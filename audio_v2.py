#!/usr/bin/env python3
"""
Meeting Copilot (macOS): системный звук → транскрипция → краткий ответ в UI

Вкладки:
1) Настройки — выбор входа (BlackHole 2ch), просмотр частоты/каналов, сохранение.
2) Звонок — запись системного звука, транскрипция, краткий ответ от лица PM/PMO.
3) О себе — ввод профиля (до 400 слов), валидация (сжатое резюме), сохранение;
   кастомный промпт для управления стилем ответов.

Хранение:
- Конфиг в ~/.meeting_copilot.json  (device_name, custom_prompt, user_profile)
- API-ключ читается из .env (рядом со скриптом) — строка OPENAI_API_KEY=sk-...

Зависимости: openai>=1.40.0, sounddevice
"""

import os
import sys
import json
import queue
import wave
import threading
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

try:
    from openai import OpenAI
except Exception:
    print("[!] Требуется пакет openai>=1.40.0:\n    pip install --upgrade openai")
    raise

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".meeting_copilot.json")
DEFAULT_DEVICE_HINTS = ["BlackHole 2ch", "BlackHole"]
PROFILE_WORD_LIMIT = 400


# -----------------------------
# Загрузка API-ключа из .env
# -----------------------------
def load_env_key() -> str | None:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        os.environ["OPENAI_API_KEY"] = key
                        return key
        except Exception as e:
            print(f"[warn] .env read error: {e}")
    return None


load_env_key()


# -----------------------------
# Конфиг приложения
# -----------------------------
class AppConfig:
    def __init__(self):
        self.device_name: str | None = None
        self.custom_prompt: str | None = None
        self.user_profile: str | None = None  # уже валидация/сжатая версия

    @classmethod
    def load(cls) -> "AppConfig":
        cfg = cls()
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cfg.device_name = data.get("device_name")
                cfg.custom_prompt = data.get("custom_prompt")
                cfg.user_profile = data.get("user_profile")
            except Exception:
                pass
        return cfg

    def save(self):
        data = {
            "device_name": self.device_name,
            "custom_prompt": self.custom_prompt,
            "user_profile": self.user_profile,
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[warn] config save error: {e}")


# -----------------------------
# Низкоуровневая запись WAV
# -----------------------------
class AudioRecorder:
    def __init__(self, device_name: str, channels: int = 2, samplerate: float | None = None):
        self.device_name = device_name
        self.channels = channels
        self.stream = None
        self.q = queue.Queue()
        self.recording = False
        self.frames_written = 0
        self.samplerate = samplerate
        self.wavefile = None
        self.filename = None
        self.max_input_channels = None

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"[sd] {status}")
        self.q.put(bytes(indata))

    def start(self, filename: str):
        if self.recording:
            return
        self.filename = filename
        dev = sd.query_devices(self.device_name, 'input')
        self.max_input_channels = int(dev.get('max_input_channels', 2) or 2)
        if self.samplerate is None:
            self.samplerate = float(dev.get('default_samplerate', 48000))

        self.wavefile = wave.open(self.filename, 'wb')
        use_channels = min(self.channels, max(1, self.max_input_channels))
        self.wavefile.setnchannels(use_channels)
        self.wavefile.setsampwidth(2)  # int16
        self.wavefile.setframerate(int(self.samplerate))
        self.frames_written = 0

        self.recording = True
        self.stream = sd.RawInputStream(
            samplerate=int(self.samplerate),
            blocksize=0,
            dtype='int16',
            channels=use_channels,
            callback=self._callback,
            device=self.device_name,
        )
        self.stream.start()
        threading.Thread(target=self._writer_thread, daemon=True).start()

    def _writer_thread(self):
        while self.recording:
            try:
                data = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            self.wavefile.writeframes(data)
            self.frames_written += len(data)

    def stop(self):
        if not self.recording:
            return None
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        while not self.q.empty():
            self.wavefile.writeframes(self.q.get())
        if self.wavefile:
            self.wavefile.close()
            self.wavefile = None
        return self.filename


# -----------------------------
# Основное приложение
# -----------------------------
class MeetingCopilotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Meeting Copilot — системный звук (BlackHole) → краткий ответ")
        self.root.geometry("980x720")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY не найден. Создайте .env рядом со скриптом.")
        self.client = OpenAI()

        self.cfg = AppConfig.load()
        self.recorder: AudioRecorder | None = None
        self.timer_running = False
        self.start_time = None
        self.processing_cancelled = False  # Флаг для отмены обработки

        self._build_ui()
        self._load_devices_into_ui()
        self._apply_config_to_ui()
        self._apply_text_config_to_ui()

    # ---------- UI ----------
    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True)
        self.tab_settings = ttk.Frame(nb)
        self.tab_call = ttk.Frame(nb)
        self.tab_about = ttk.Frame(nb)
        nb.add(self.tab_settings, text="Настройки")
        nb.add(self.tab_call, text="Звонок")
        nb.add(self.tab_about, text="О себе")

        # ===== Настройки =====
        s = ttk.Frame(self.tab_settings, padding=12)
        s.pack(fill=tk.BOTH, expand=True)

        r1 = ttk.Frame(s); r1.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(r1, text="Вход (источник записи)").pack(side=tk.LEFT)
        self.device_box = ttk.Combobox(r1, state="readonly", width=60)
        self.device_box.pack(side=tk.LEFT, padx=8)
        ttk.Button(r1, text="Обновить", command=self._load_devices_into_ui).pack(side=tk.LEFT)

        r2 = ttk.Frame(s); r2.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(r2, text="Частота / Каналы:").pack(side=tk.LEFT)
        self.device_meta = tk.StringVar(value="—")
        ttk.Label(r2, textvariable=self.device_meta).pack(side=tk.LEFT, padx=8)

        r3 = ttk.Frame(s); r3.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(r3, text="Сохранить устройство", command=self._save_device).pack(side=tk.LEFT)
        ttk.Button(r3, text="Список устройств в консоль", command=self._print_devices_to_console).pack(side=tk.LEFT, padx=8)

        self.settings_status = tk.StringVar(value="Выбери 'BlackHole 2ch' как вход.")
        ttk.Label(s, textvariable=self.settings_status).pack(fill=tk.X, pady=(12, 0))

        # ===== Звонок =====
        c = ttk.Frame(self.tab_call, padding=12)
        c.pack(fill=tk.BOTH, expand=True)

        btn = ttk.Frame(c); btn.pack(fill=tk.X)
        self.start_btn = ttk.Button(btn, text="Начать запись", command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btn, text="Остановить и проанализировать", command=self.stop_and_process, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        self.cancel_btn = ttk.Button(btn, text="Отменить", command=self.cancel_recording, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Готово. Вкладка 'О себе' и промпт будут учтены.")
        ttk.Label(c, textvariable=self.status_var).pack(fill=tk.X, pady=(8, 8))

        ttk.Label(c, text="Ответ (краткий + подробный + пример при наличии):").pack(anchor=tk.W)
        self.answer_box = ScrolledText(c, height=18, wrap=tk.WORD)
        self.answer_box.pack(fill=tk.BOTH, expand=True, pady=(2, 10))

        ttk.Label(c, text="Журнал:").pack(anchor=tk.W)
        self.log_box = ScrolledText(c, height=12, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # ===== О себе =====
        a = ttk.Frame(self.tab_about, padding=12)
        a.pack(fill=tk.BOTH, expand=True)

        # Кастомный промпт
        ttk.Label(a, text="Кастомный промпт (используется при генерации ответа):").pack(anchor=tk.W)
        self.prompt_box = ScrolledText(a, height=8, wrap=tk.WORD)
        self.prompt_box.pack(fill=tk.BOTH, expand=False, pady=(2, 8))
        prompt_hint = (
            "Пример промпта:\n"
            "Ты — Senior PM/PMO на интервью. Структура ответа:\n"
            "1) Краткий ответ (2-3 предл.) — суть решения\n"
            "2) Подробно (3-5 предл.) — обоснование, trade-offs\n"
            "3) Пример из практики (ОПЦИОНАЛЬНО) — только при наличии релевантного кейса из топ-компании\n"
            "Адаптируйся к типу вопроса. Frameworks — только когда релевантны. Не придумывай примеры."
        )
        ttk.Label(a, text=prompt_hint, foreground="#888").pack(anchor=tk.W, pady=(0, 8))
        ttk.Button(a, text="Сохранить промпт", command=self._save_prompt).pack(anchor=tk.W)

        ttk.Separator(a, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        # Профиль
        ttk.Label(a, text=f"О себе (до {PROFILE_WORD_LIMIT} слов):").pack(anchor=tk.W)
        self.profile_box = ScrolledText(a, height=10, wrap=tk.WORD)
        self.profile_box.pack(fill=tk.BOTH, expand=True, pady=(2, 4))
        self.word_count_var = tk.StringVar(value="0 слов")
        ttk.Label(a, textvariable=self.word_count_var, foreground="#888").pack(anchor=tk.W)

        abtn = ttk.Frame(a); abtn.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(abtn, text="Валидировать (сжать и структурировать)", command=self._validate_profile).pack(side=tk.LEFT)
        ttk.Button(abtn, text="Сохранить профиль", command=self._save_profile).pack(side=tk.LEFT, padx=8)
        ttk.Button(abtn, text="Переписать", command=self._rewrite_profile).pack(side=tk.LEFT)

        ttk.Label(a, text="Результат валидации (краткое резюме):").pack(anchor=tk.W, pady=(8, 2))
        self.profile_validated = ScrolledText(a, height=6, wrap=tk.WORD)
        self.profile_validated.pack(fill=tk.BOTH, expand=False)

        # автообновление счетчика слов
        self.profile_box.bind("<<Modified>>", self._on_profile_modified)
        self.device_box.bind("<<ComboboxSelected>>", lambda e: self._update_device_meta())

    # ---------- Настройки устройств ----------
    def _load_devices_into_ui(self):
        devices = sd.query_devices()
        input_devices = []
        for d in devices:
            if d.get('max_input_channels', 0) > 0:
                input_devices.append(d['name'])
        input_devices.sort(key=lambda n: (0 if any(h in n for h in DEFAULT_DEVICE_HINTS) else 1, n))
        self.device_box['values'] = input_devices
        if not input_devices:
            self.settings_status.set("Не найдено входных устройств. Установите BlackHole 2ch.")
        else:
            self.settings_status.set("Выбери входное устройство.")
        self._update_device_meta()

    def _apply_config_to_ui(self):
        values = list(self.device_box['values'])
        sel_idx = None
        if self.cfg.device_name and self.cfg.device_name in values:
            sel_idx = values.index(self.cfg.device_name)
        if sel_idx is None:
            for i, name in enumerate(values):
                if 'BlackHole 2ch' in name:
                    sel_idx = i; break
        if sel_idx is None and values:
            sel_idx = 0
        if sel_idx is not None:
            self.device_box.current(sel_idx)
            self._update_device_meta()

    def _update_device_meta(self):
        name = self.device_box.get()
        if not name:
            self.device_meta.set("—"); return
        try:
            d = sd.query_devices(name, 'input')
            sr = int(d.get('default_samplerate', 0))
            ch = int(d.get('max_input_channels', 0))
            self.device_meta.set(f"{name} — sr:{sr} Hz, ch:{ch}")
        except Exception as e:
            self.device_meta.set(f"{name} — параметры недоступны: {e}")

    def _save_device(self):
        name = self.device_box.get()
        if not name:
            self.settings_status.set("Сначала выберите устройство."); return
        self.cfg.device_name = name
        self.cfg.save()
        self.settings_status.set(f"Сохранено: {name}")

    def _print_devices_to_console(self):
        print("==== INPUT devices ====")
        for d in sd.query_devices():
            if d.get('max_input_channels', 0) > 0:
                print(f"IN  : {d['name']} | sr:{int(d.get('default_samplerate', 0))} | ch:{d.get('max_input_channels')}")
        print("==== OUTPUT devices ====")
        for d in sd.query_devices():
            if d.get('max_output_channels', 0) > 0:
                print(f"OUT : {d['name']} | sr:{int(d.get('default_samplerate', 0))} | ch:{d.get('max_output_channels')}")
        self.settings_status.set("Список устройств напечатан в консоль.")

    # ---------- Настройки текста (промпт/профиль) ----------
    def _apply_text_config_to_ui(self):
        # заполнить UI из конфигурации
        if self.cfg.custom_prompt:
            self.prompt_box.delete("1.0", tk.END)
            self.prompt_box.insert(tk.END, self.cfg.custom_prompt)
        if self.cfg.user_profile:
            self.profile_validated.delete("1.0", tk.END)
            self.profile_validated.insert(tk.END, self.cfg.user_profile)

    def _save_prompt(self):
        text = self.prompt_box.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Промпт", "Промпт пустой — ничего не сохранено."); return
        self.cfg.custom_prompt = text
        self.cfg.save()
        messagebox.showinfo("Промпт", "Промпт сохранён. Он будет использован при ответе.")

    def _on_profile_modified(self, _event=None):
        try:
            self.profile_box.edit_modified(False)
            words = len(self.profile_box.get("1.0", tk.END).strip().split())
            self.word_count_var.set(f"{words} слов")
        except Exception:
            pass

    def _word_limit_ok(self, txt: str) -> bool:
        return len(txt.strip().split()) <= PROFILE_WORD_LIMIT

    def _validate_profile(self):
        raw = self.profile_box.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showwarning("О себе", "Введите текст профиля."); return
        if not self._word_limit_ok(raw):
            messagebox.showwarning("О себе", f"Сократите профиль до {PROFILE_WORD_LIMIT} слов."); return

        self.status_var.set("Валидация профиля…")
        threading.Thread(target=self._validate_profile_thread, args=(raw,), daemon=True).start()

    def _validate_profile_thread(self, raw: str):
        try:
            # короткое структурирование профиля
            prompt = (
                "Сожми и структурируй следующий текст профиля кандидата PM/PMO. "
                "Верни 3–4 кратких предложения: компетенции, индустрии/продукты, ключевые достижения, инструменты. "
                "Без воды, без маркеров.\n\n"
                f"Текст:\n{raw}"
            )
            cmp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "Ты карьерный коуч. Пиши кратко и по делу."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=220,
            )
            summary = cmp.choices[0].message.content.strip()
            self.profile_validated.delete("1.0", tk.END)
            self.profile_validated.insert(tk.END, summary)
            self.status_var.set("Профиль валидирован. Нажмите «Сохранить профиль», если ок.")
        except Exception as e:
            self.profile_validated.delete("1.0", tk.END)
            self.profile_validated.insert(tk.END, f"Ошибка валидации: {e}")
            self.status_var.set("Ошибка валидации.")

    def _save_profile(self):
        text = self.profile_validated.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("О себе", "Нет результата валидации для сохранения."); return
        self.cfg.user_profile = text
        self.cfg.save()
        messagebox.showinfo("О себе", "Профиль сохранён. Он будет учитываться в ответах.")

    def _rewrite_profile(self):
        self.profile_box.delete("1.0", tk.END)
        self.profile_validated.delete("1.0", tk.END)
        self.word_count_var.set("0 слов")
        messagebox.showinfo("О себе", "Введите новый текст и снова нажмите «Валидировать».")

    # ---------- Лог ----------
    def log(self, text: str):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_box.insert(tk.END, f"[{ts}] {text}\n")
        self.log_box.see(tk.END)

    # ---------- Запись и обработка ----------
    def start_recording(self):
        device = self.device_box.get() or self.cfg.device_name
        if not device:
            self.status_var.set("Выберите устройство на вкладке 'Настройки'."); return
        self.cfg.device_name = device
        self.cfg.save()

        self.answer_box.delete('1.0', tk.END)
        self.processing_cancelled = False  # Сброс флага при новой записи
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.cancel_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Запись… Вход: {device}.")
        self.log(f"Начата запись с входа: {device}")

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_file = os.path.join(os.getcwd(), f"capture_{ts}.wav")

        dev = sd.query_devices(device, 'input')
        # Оптимизация для речи: 16kHz mono (отличное качество, малый размер файла)
        # 10 минут = ~19 MB (влезает в Whisper API лимит 25 MB)
        sr = 16000  # 16 kHz достаточно для речи (стандарт телефонии)
        ch = 1      # Mono - речь не требует стерео

        self.recorder = AudioRecorder(device_name=device, channels=ch, samplerate=sr)
        self.recorder.start(self.out_file)
        self.start_time = datetime.now()
        self.timer_running = True
        self._tick_timer()

    def _tick_timer(self):
        if not self.timer_running or self.start_time is None:
            return
        sec = int((datetime.now() - self.start_time).total_seconds())
        self.status_var.set(f"Запись… {sec} c.")
        self.root.after(500, self._tick_timer)

    def stop_and_process(self):
        self.stop_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.timer_running = False
        self.status_var.set("Остановка…")
        if self.recorder:
            wav_path = self.recorder.stop()
            self.log(f"Файл записан: {wav_path}")
            self.status_var.set("Расшифровка и генерация ответа…")
            threading.Thread(target=self._transcribe_and_answer, args=(wav_path,), daemon=True).start()
        else:
            self.status_var.set("Не было активной записи.")
            self.start_btn.configure(state=tk.NORMAL)

    def cancel_recording(self):
        """Отменяет текущую запись без обработки"""
        self.processing_cancelled = True
        self.timer_running = False

        if self.recorder:
            wav_path = self.recorder.stop()
            self.log(f"Запись отменена. Файл удалён: {wav_path}")
            # Удаляем файл, так как он не нужен
            try:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception as e:
                self.log(f"Не удалось удалить файл: {e}")

        # Возвращаем UI в исходное состояние
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.status_var.set("Запись отменена. Готово к новой записи.")
        self.processing_cancelled = False

    def _compose_prompt(self, question_text: str) -> str:
        """
        Собираем промпт:
        - если задан кастомный промпт в настройках — используем его как вводную инструкцию;
        - добавляем сам вопрос (транскрипт);
        - если сохранён профиль — приклеиваем как контекст (модель может сослаться на релевантный опыт).
        """
        base = self.cfg.custom_prompt.strip() if self.cfg.custom_prompt else (
            "Ты — опытный Senior Product Manager / Senior Project Manager на интервью. "
            "Отвечай от первого лица, адаптируя подход к типу вопроса:\n"
            "- Стратегические вопросы → бизнес-цели, метрики, приоритизация\n"
            "- Тактические вопросы → конкретные шаги, инструменты, процессы\n"
            "- Конфликтные ситуации → stakeholder management, компромиссы\n"
            "- Поведенческие вопросы → конкретные примеры из опыта\n\n"
            "Используй frameworks (RICE, Jobs-to-be-Done, OKR) ТОЛЬКО когда они естественно подходят к вопросу."
        )
        prompt = (
            f"{base}\n\n"
            "Ниже транскрипт вопроса с собеседования. "
            "Проанализируй тип вопроса и дай релевантный ответ:\n\n"
            "**Краткий ответ** (2-3 предложения) — прямой ответ на вопрос\n\n"
            "**Подробно** (3-5 предложений) — обоснование, trade-offs или next steps\n\n"
            "**Пример из практики** (ОПЦИОНАЛЬНО, 2-3 предложения) — добавь ТОЛЬКО если знаешь "
            "конкретный релевантный кейс из топ-компании (Netflix, Amazon, Spotify, Google, Airbnb, Tesla и т.д.) "
            "с реальными метриками. Лучше БЕЗ примера, чем с натянутым или неточным. "
            "Не придумывай примеры — используй только известные публичные кейсы.\n\n"
            f"Вопрос (транскрипт):\n{question_text}"
        )
        if self.cfg.user_profile:
            prompt += f"\n\nДополнительный контекст о мне (учитывай релевантный опыт, можно ссылаться кратко):\n{self.cfg.user_profile}"
        return prompt

    def _transcribe_and_answer(self, wav_path: str):
        try:
            # Проверка отмены перед началом обработки
            if self.processing_cancelled:
                self.log("Обработка отменена пользователем.")
                return

            # 1) Транскрипция
            with open(wav_path, 'rb') as f:
                tr = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    temperature=0.0,
                    language="ru",
                )
            text = tr.text if hasattr(tr, 'text') else getattr(tr, 'text', '')
            self.log(f"Транскрипт ({len(text)} симв.)")
            if not text.strip():
                raise RuntimeError("Пустой транскрипт. Проверьте источник звука/громкость.")

            # Проверка отмены перед генерацией ответа
            if self.processing_cancelled:
                self.log("Обработка отменена после транскрипции.")
                return

            # 2) Краткий ответ (PM/PO; учитываем профиль и кастом-промпт)
            full_prompt = self._compose_prompt(text)
            cmp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "Ты — Senior Product Manager / Senior Project Manager с 7+ годами опыта на интервью.\n\n"
                        "ПРИНЦИПЫ ОТВЕТА:\n"
                        "1. Адаптируй подход к типу вопроса — не используй шаблоны\n"
                        "2. Frameworks (RICE, Jobs-to-be-Done, OKR) — только если релевантны контексту\n"
                        "3. Для стратегических вопросов — фокус на бизнес-целях и метриках\n"
                        "4. Для тактических — конкретные процессы и инструменты\n"
                        "5. Для конфликтов — stakeholder management и компромиссы\n"
                        "6. Для поведенческих — реальные примеры и выводы\n\n"
                        "ФОРМАТ ОТВЕТА (2-3 блока):\n\n"
                        "**Краткий ответ** (2-3 предложения, ~50-70 слов):\n"
                        "[Прямой ответ на вопрос с ключевым решением]\n\n"
                        "**Подробно** (3-5 предложений, ~100-130 слов):\n"
                        "[Контекст, обоснование или примеры — в зависимости от типа вопроса. "
                        "Используй frameworks только если они добавляют ценность.]\n\n"
                        "**Пример из практики** (ОПЦИОНАЛЬНО, 2-3 предложения, ~60-80 слов):\n"
                        "[Добавляй ТОЛЬКО если знаешь конкретный релевантный кейс из топ-компании "
                        "(Netflix, Amazon, Spotify, Google, Airbnb, Tesla, Uber, etc.) с реальными публичными метриками. "
                        "Укажи: (1) Компанию, (2) Что сделали, (3) Измеримый результат. "
                        "ВАЖНО: Лучше БЕЗ примера, чем с натянутым или неточным. Не придумывай примеры.]\n\n"
                        "СТИЛЬ: Уверенный senior-level, конкретный, без клише и воды."
                    )},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.3,  # Немного повышено для более естественных ответов
                max_tokens=800,  # Для 2-3 блоков (пример опциональный)
            )
            answer = cmp.choices[0].message.content.strip()
            self._append_answer(answer)
            self.status_var.set("Готово. Можно начать новую запись.")
        except Exception as e:
            msg = str(e)
            if "insufficient_quota" in msg or "exceeded your current quota" in msg:
                self._append_answer(
                    "Нет квоты API. Зайдите на platform.openai.com → Billing, пополните баланс/лимиты и повторите."
                )
            else:
                self._append_answer(f"Ошибка: {e}")
        finally:
            self.start_btn.configure(state=tk.NORMAL)

    def _append_answer(self, text: str):
        self.answer_box.delete('1.0', tk.END)
        self.answer_box.insert(tk.END, text)
        self.answer_box.see(tk.END)


def main():
    try:
        root = tk.Tk()
        app = MeetingCopilotApp(root)
        root.mainloop()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
