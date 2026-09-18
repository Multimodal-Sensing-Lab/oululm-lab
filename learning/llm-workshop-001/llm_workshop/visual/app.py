"""Tk desktop UI. All learning computations are in the numbered demo.py files."""
from pathlib import Path
import os
import tempfile
# Keep matplotlib caches writable without changing the user's home configuration.
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'llm-workshop-matplotlib'))
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
import torch
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from llm_workshop.teaching import Session, TOPICS, topic

TITLES = ['The whole pipeline', 'Text → tokens → IDs', 'Embeddings + positions', 'Inside a transformer',
          'Train from scratch', 'Generate one token', 'Adapt with LoRA', 'Retrieve evidence (RAG)', 'Compare and evaluate']
NOTES = [
    'Select a box. Training and inference share the forward path but take different branches at the end.',
    'Type in the box above. The vocabulary stays fixed. Colors identify token positions; repeated letters have the same ID.',
    'Real model values. Repeated characters share a token embedding; their positions add different vectors. Untrained vectors have no learned meaning.',
    'Real values from the selected block/head. Attention rows read allowed earlier/current positions. FFN and residual views show all heads combined.',
    'CPU training on separate story documents. Live curves use fixed train/validation windows. The shared model changes only when you start training.',
    'Generate explicitly, one ID at a time. Temperature changes probabilities; greedy selection ignores temperature. Neither changes weights.',
    'Tiny character-model LoRA mechanics: only the output-head A/B matrices learn. This is not pretrained instruction-model SFT or a QLoRA simulation.',
    'One passage per line. Edit a room number and watch the evidence in the prompt change. Retrieval is real TF-IDF; this window does not fabricate an LLM answer.',
    'Metric fixture: labels below are manually authored examples. Load two checkpoints for real generations on the same prompt and greedy decoding.',
]
COLORS = ['#dcecff', '#efe2ff', '#d8f4e6', '#ffedcb', '#fce0e5', '#d7f3f4']


class Lab(tk.Tk):
    def __init__(self, initial_topic=0):
        super().__init__()
        self.title('OuluLLM Lab · language model laboratory')
        self.geometry('1260x920')
        self.minsize(1020, 760)
        self.configure(bg='#edf2f8')
        self.session = Session()
        self.training = None
        self.adapter = None
        self.generation = None
        self.timer = None
        self.debounce = None
        self.current = initial_topic
        self.text = tk.StringVar(value='the cat sat.')
        self.status = tk.StringVar()
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#edf2f8')
        self.style.configure('TLabel', background='#edf2f8', foreground='#162d49', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 10), padding=6)
        header = tk.Frame(self, bg='#152d4c', padx=20, pady=14)
        header.pack(fill='x')
        tk.Label(header, text='OuluLLM Lab  /  Open the model', fg='white', bg='#152d4c', font=('Helvetica', 21, 'bold')).pack(anchor='w')
        tk.Label(header, text='CPU classroom lab · real tensors · separate code and presentation', fg='#c6d8f0', bg='#152d4c', font=('Helvetica', 11)).pack(anchor='w')
        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        nav = ttk.Frame(body, width=205)
        nav.pack(side='left', fill='y', padx=(0, 12))
        for i, name in enumerate(TITLES):
            ttk.Button(nav, text=f'{i:03d}  {name}', command=lambda i=i: self.select(i)).pack(fill='x', pady=3)
        ttk.Separator(nav).pack(fill='x', pady=14)
        ttk.Button(nav, text='Open topic instructions', command=self.open_notes).pack(fill='x', pady=3)
        ttk.Button(nav, text='Step-by-step walkthrough', command=self.open_walkthrough).pack(fill='x', pady=3)
        ttk.Button(nav, text='Load tiny checkpoint', command=self.load_model).pack(fill='x', pady=3)
        ttk.Button(nav, text='Reset shared model', command=self.reset_model).pack(fill='x', pady=3)
        self.model_label = tk.Label(nav, wraplength=190, justify='left', bg='#edf2f8', fg='#66438e', font=('Helvetica', 10, 'bold'))
        self.model_label.pack(fill='x', pady=16)
        main = ttk.Frame(body)
        main.pack(side='left', fill='both', expand=True)
        self.heading = ttk.Label(main, font=('Helvetica', 21, 'bold'))
        self.heading.pack(anchor='w')
        self.note = tk.Label(main, wraplength=880, justify='left', bg='#e0ecfa', fg='#1c416b', padx=12, pady=10, font=('Helvetica', 11))
        self.note.pack(fill='x', pady=8)
        main.bind('<Configure>', lambda event: self.note.configure(wraplength=max(300, event.width-28)))
        row = ttk.Frame(main)
        row.pack(fill='x')
        ttk.Label(row, text='Input text / prompt').pack(side='left', padx=(0, 10))
        self.entry = ttk.Entry(row, textvariable=self.text, font=('Courier', 13))
        self.entry.pack(side='left', fill='x', expand=True)
        self.text.trace_add('write', self.schedule_render)
        self.controls = ttk.Frame(main, padding=(0, 8))
        self.controls.pack(fill='x')
        self.content = ttk.Frame(main)
        self.content.pack(fill='both', expand=True)
        self.footer = ttk.Label(main, textvariable=self.status, wraplength=900)
        self.footer.pack(fill='x', pady=8)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.select(initial_topic)

    def schedule_render(self, *_):
        if self.debounce:
            self.after_cancel(self.debounce)
        self.debounce = self.after(250, self.refresh_input)

    def refresh_input(self):
        self.debounce = None
        self.generation = None
        self.render()

    def stop(self):
        if self.timer:
            self.after_cancel(self.timer)
            self.timer = None
        if self.current == 4 and self.training:
            if self.training.history[-1]['step'] != self.training.step:
                self.training.history.append(self.training.measure())
        self.status.set('Paused. Parameters and current results are retained.')

    def close(self):
        self.stop()
        self.destroy()

    def reset_model(self):
        self.stop()
        self.session = Session()
        self.training = self.adapter = self.generation = None
        self.select(self.current)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[('TinyGPT checkpoint', '*.pt')])
        if path:
            self.stop()
            try:
                candidate = Session()
                candidate.load(path)
                self.session = candidate
                self.training = self.adapter = self.generation = None
                self.select(self.current)
            except Exception as exc:
                messagebox.showerror('Checkpoint could not be loaded', str(exc))

    def open_notes(self):
        folder = next(p for p in TOPICS if p.name.startswith(f'{self.current:03d}_'))
        webbrowser.open((folder/'index.html').as_uri())

    def open_walkthrough(self):
        folder = next(p for p in TOPICS if p.name.startswith('001_'))
        anchor = {1: 'tokens', 2: 'embeddings', 3: 'attention'}.get(self.current, 'start')
        webbrowser.open((folder/'walkthrough.html').as_uri() + '#' + anchor)

    def button(self, text, command):
        ttk.Button(self.controls, text=text, command=command).pack(side='left', padx=(0, 8))

    def combo(self, label, values, default):
        ttk.Label(self.controls, text=label).pack(side='left', padx=(0, 4))
        variable = tk.StringVar(value=str(default))
        box = ttk.Combobox(self.controls, textvariable=variable, values=list(map(str, values)), state='readonly', width=15)
        box.pack(side='left', padx=(0, 12))
        box.bind('<<ComboboxSelected>>', lambda _: self.render())
        return variable

    def output_box(self, height=9):
        box = tk.Text(self.content, height=height, wrap='word', font=('Courier', 11), bg='#fff', fg='#19314e', relief='flat', padx=12, pady=10)
        scroll = ttk.Scrollbar(self.content, command=box.yview)
        box.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        box.pack(fill='both', expand=True)
        return box

    def plot(self):
        self.figure = Figure(figsize=(9, 4), dpi=100, layout='constrained', facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.content)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=(0, 8))

    def select(self, number):
        self.stop()
        if self.debounce:
            self.after_cancel(self.debounce)
            self.debounce = None
        self.current = number
        for container in (self.controls, self.content):
            for widget in container.winfo_children():
                widget.destroy()
        self.figure = self.canvas = self.output = None
        self.heading.configure(text=f'{number:03d}  {TITLES[number]}')
        self.note.configure(text=NOTES[number])
        self.model_label.configure(text=self.session.label)
        if number == 0:
            self.pipeline = tk.Canvas(self.content, bg='white', highlightthickness=0, height=440)
            self.pipeline.pack(fill='both', expand=True)
            self.output = self.output_box(4)
            self.pipeline.bind('<Configure>', lambda _: self.draw_pipeline())
        elif number == 1:
            self.stage = self.combo('Reveal through', ['Text', 'Pieces', 'Vocabulary', 'IDs', 'Decode'], 'Decode')
            self.button('Next stage', self.next_stage)
            self.tiles = tk.Canvas(self.content, bg='white', height=165, highlightthickness=0)
            self.tiles.pack(fill='x')
            self.tiles.bind('<Configure>', lambda _: self.render())
            self.output = self.output_box()
        elif number == 2:
            self.plot()
            self.output = self.output_box(5)
        elif number == 3:
            self.block = self.combo('Block', range(self.session.model.cfg.n_layer), 0)
            self.head = self.combo('Head', range(self.session.model.cfg.n_head), 0)
            self.view = self.combo('View', ['attention', 'scores', 'masked_scores', 'Q', 'K', 'V', 'mask', 'weighted_values', 'input', 'after_attention_residual', 'feed_forward', 'output'], 'attention')
            self.plot()
            self.output = self.output_box(5)
        elif number == 4:
            self.updates = self.combo('Updates', [10, 30, 100, 200, 500], 100)
            self.rate = self.combo('Learning rate', [.0003, .001, .003, .01], .003)
            self.button('Start / continue', self.start_training)
            self.button('Stop', self.stop)
            self.button('Save run', self.save_training)
            self.plot()
            self.output = self.output_box(5)
        elif number == 5:
            self.temperature = self.combo('Temperature', [.2, .5, .8, 1., 1.5, 2.], .8)
            self.greedy = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.controls, text='Greedy', variable=self.greedy).pack(side='left')
            self.button('Next token', self.generate_one)
            self.button('Reset prompt', self.reset_generation)
            self.plot()
            self.output = self.output_box(5)
        elif number == 6:
            self.domain = self.combo('Domain', ['biology', 'email', 'finance'], 'biology')
            self.rank = self.combo('Rank', [1, 2, 4, 8], 4)
            self.button('New adapter + 60 steps', self.start_adapter)
            self.button('Stop', self.stop)
            self.button('Save adapter', self.save_adapter)
            self.plot()
            self.output = self.output_box(6)
        elif number == 7:
            self.text.set(topic(7).QUESTION)
            ttk.Label(self.controls, text='Edit evidence below: one passage per line').pack(side='left')
            self.docs = tk.Text(self.content, height=5, wrap='word', font=('Courier', 11))
            self.docs.insert('1.0', topic(7).DOCUMENTS)
            self.docs.pack(fill='x', pady=6)
            self.docs.bind('<KeyRelease>', self.schedule_render)
            self.plot()
            self.output = self.output_box(8)
        elif number == 8:
            self.button('Compare two checkpoints', self.compare_checkpoints)
            self.predictions = tk.Text(self.content, height=6, font=('Courier', 11))
            self.predictions.insert('1.0', '\n'.join(topic(8).PREDICTIONS))
            self.predictions.pack(fill='x', pady=6)
            self.predictions.bind('<KeyRelease>', self.schedule_render)
            self.plot()
            self.output = self.output_box(6)
        if number != 7 and self.text.get() == topic(7).QUESTION:
            self.text.set('the cat sat.')
        self.render()

    def write(self, text):
        if self.output:
            self.output.delete('1.0', 'end')
            self.output.insert('1.0', text)

    def render(self):
        try:
            self.model_label.configure(text=self.session.label)
            if self.figure:
                self.figure.clear()
            method = getattr(self, f'render_{self.current}')
            method()
            if self.canvas:
                self.canvas.draw_idle()
            self.status.set('Ready. Input edits update the view after a short pause.')
        except (ValueError, RuntimeError, IndexError) as exc:
            self.status.set(str(exc))
            self.write(str(exc))
            if self.figure:
                self.figure.clear()
                self.canvas.draw_idle()

    def draw_pipeline(self):
        self.pipeline.delete('all')
        w = max(750, self.pipeline.winfo_width())
        data = topic(0).STAGES
        for i, (title, detail) in enumerate(data):
            x, y = (25 if i < 5 else w/2 + 30), (18 + i*72 if i < 5 else 105 + (i-5)*170)
            width = w/2-60
            fill = COLORS[i % len(COLORS)]
            tag = f'stage{i}'
            self.pipeline.create_rectangle(x, y, x+width, y+55, fill=fill, outline='#879cb5', width=2, tags=tag)
            self.pipeline.create_text(x+15, y+27, anchor='w', text=f'{i+1:02d}  {title}', font=('Helvetica', 13, 'bold'), tags=tag)
            self.pipeline.tag_bind(tag, '<Button-1>', lambda event, t=title, d=detail: self.write(f'{t}\n\n{d}'))
            if i < 4:
                self.pipeline.create_line(x+width/2, y+55, x+width/2, y+72, arrow='last', fill='#405d7e', width=2)
        self.pipeline.create_text(w*.74, 40, text='Choose a branch after the forward path', font=('Helvetica', 11), width=w/2-60)
        self.pipeline.create_line(w/2-35, 261, w/2+12, 261, w/2+12, 145, w/2+30, 145, arrow='last', fill='#227450', width=2)
        self.pipeline.create_line(w/2-35, 333, w/2+20, 333, w/2+20, 315, w/2+30, 315, arrow='last', fill='#9a6200', width=2)

    def render_0(self):
        self.draw_pipeline()
        self.write('Click a component to reveal its role.\n\nTraining uses logits + known targets for cross-entropy (softmax is handled inside the loss). Inference applies a selection policy and repeats with the new token.\n\nThe full block diagram and credited architecture images are in “Open topic instructions”.')

    def next_stage(self):
        stages = ['Text', 'Pieces', 'Vocabulary', 'IDs', 'Decode']
        self.stage.set(stages[(stages.index(self.stage.get())+1) % len(stages)])
        self.render()

    def render_1(self):
        text = self.text.get()
        if len(text) > 64:
            raise ValueError('Use up to 64 characters so the tiles remain readable.')
        self.tiles.delete('all')
        stage = ['Text', 'Pieces', 'Vocabulary', 'IDs', 'Decode'].index(self.stage.get())
        width = max(650, self.tiles.winfo_width())
        columns = max(1, int((width-20)/62))
        self.tiles.configure(height=max(80, ((len(text)+columns-1)//columns)*62+15))
        for i, char in enumerate(text):
            x, y = 10+(i % columns)*62, 10+(i//columns)*62
            known = char in self.session.stoi
            self.tiles.create_rectangle(x, y, x+56, y+54, fill=COLORS[i % len(COLORS)] if known else '#ffcbd0', outline='#b2c3d5')
            self.tiles.create_text(x+28, y+16, text=repr(char) if stage else 'text', font=('Courier', 11, 'bold'))
            label = f'ID {self.session.stoi[char]}' if known else 'unknown'
            self.tiles.create_text(x+28, y+37, text=label if stage >= 3 else f'pos {i}', font=('Helvetica', 9))
        lines = [f'INPUT: {text!r}']
        if stage >= 1:
            lines.append(f'PIECES: {list(text)!r}')
        if stage >= 2:
            lines.append('FIXED VOCABULARY:\n'+'  '.join(f'{i}:{char!r}' for i, char in enumerate(self.session.vocabulary)))
        if stage >= 3:
            unknown = sorted(set(text)-set(self.session.stoi))
            if unknown:
                lines.append(f'Unsupported characters: {unknown!r}. No replacement or model update was made.')
            else:
                ids = [self.session.stoi[c] for c in text]
                lines.append(f'IDS: {ids}')
                if stage >= 4:
                    lines.append(f'DECODED: {self.session.decode(ids)!r}')
        self.write('\n\n'.join(lines))

    def heatmap(self, ax, values, title, labels=None, probabilities=False,
                limit=None, xlabel='Feature column', ylabel='Token position',
                color_label='Value'):
        array = np.asarray(values, dtype=float)
        masked = np.ma.masked_invalid(array)
        if not probabilities and limit is None:
            finite = array[np.isfinite(array)]
            limit = max(float(np.abs(finite).max()), 1e-6) if finite.size else 1.
        image = ax.imshow(masked, aspect='auto', cmap='Blues' if probabilities else 'coolwarm',
                          vmin=0 if probabilities else -limit, vmax=1 if probabilities else limit)
        ax.set_title(title, fontsize=11)
        if labels and len(labels) <= 24:
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        self.figure.colorbar(image, ax=ax, fraction=.035, label=color_label)

    def render_2(self):
        result = topic(2).inspect(self.session, self.text.get())
        labels = [f'{i}:{c!r}' for i, c in enumerate(self.text.get())]
        keys = ['tokens', 'positions', 'combined']
        shown = min(8, self.session.model.cfg.n_embd)
        limit = max(max(float(result[key][:, :shown].abs().max()) for key in keys), 1e-6)
        for ax, key in zip(self.figure.subplots(1, 3), keys):
            self.heatmap(ax, result[key][:, :shown], key, labels, limit=limit)
        self.write(f'Each row is one input position; columns are learned numerical features. Showing {shown} of {self.session.model.cfg.n_embd}.\n'
                   'Blue = negative, pale = near zero, red = positive. All three panels share a scale; read the colorbar (it can change on refresh).\n'
                   'Token IDs select rows in the token table; position numbers select rows in a separate position table. Add matching cells to enter block 0.\n'
                   f'Example, position 0 / feature 0: {result["tokens"][0, 0]:.4f} + {result["positions"][0, 0]:.4f} = {result["combined"][0, 0]:.4f} (rounded).\n'+
                   '\n'.join(f'{name}: {tuple(matrix.shape)}' for name, matrix in result.items())+
                   '\nOpen “Step-by-step walkthrough” for how the tables and colors are constructed.')

    def render_3(self):
        result = topic(3).inspect(self.session, self.text.get(), int(self.block.get()), int(self.head.get()))
        key = self.view.get()
        matrix = result[key]
        ax = self.figure.add_subplot()
        pairwise = key in {'scores', 'masked_scores', 'attention', 'mask'}
        selected_head = key in {'Q', 'K', 'V', 'scores', 'masked_scores', 'attention', 'weighted_values'}
        scope = f'head {self.head.get()}' if selected_head else ('shared mask' if key == 'mask' else 'all heads')
        explanations = {
            'input': 'The 64-feature default input is token + position in block 0, or the preceding block output in later blocks.',
            'Q': 'Query: normalize each input row, then apply a learned linear projection. Each row describes what this position can match.',
            'K': 'Key: a separate learned projection of the normalized input. Query i is compared with key j to form score[i, j].',
            'V': 'Value: a separate learned projection of the normalized input. Attention weights mix these rows to produce weighted_values.',
            'scores': 'Raw compatibility: score[i, j] = dot(Q[i], K[j]) / sqrt(head width). Higher means a stronger relative match; these are not probabilities.',
            'mask': '1 = allowed (current or earlier position); 0 = blocked (future). This fixed rule is the same for every head.',
            'masked_scores': 'Keep allowed scores; replace future scores with negative infinity, drawn blank. This makes their softmax weights exactly zero.',
            'attention': 'Apply softmax across each masked score row: exponentiate and divide by the row total. Each row distributes weight over allowed positions.',
            'weighted_values': 'For each query row, multiply every value row by its attention weight and add. This is one head’s mixture of information.',
            'after_attention_residual': 'Concatenate all head mixtures, apply the learned output projection, then add the original block input.',
            'feed_forward': 'Normalize the first residual, expand features (64 → 256 by default), apply GELU, then project back to 64. Each position is processed separately.',
            'output': 'Add the feed-forward update to the first residual. Pass this result to the next block, or to final normalization and vocabulary scores.',
        }
        self.heatmap(ax, matrix, f'Block {self.block.get()} · {scope} · {key}',
                     [f'{i}:{c!r}' for i, c in enumerate(self.text.get())], key in {'attention', 'mask'},
                     xlabel='Key position (source)' if pairwise else 'Feature column',
                     ylabel='Query position (reader)' if pairwise else 'Token position',
                     color_label='Weight' if key == 'attention' else ('Allowed' if key == 'mask' else 'Value'))
        columns = 'key positions' if pairwise else 'features'
        legend = ('White = 0; darker blue = closer to 1.' if key in {'attention', 'mask'} else
                  'Blue = negative; pale = near zero; red = positive. Blank = masked −infinity. Read the colorbar; its range can change.')
        self.write(f'{key}: {tuple(matrix.shape)} = {matrix.shape[0]} input positions × {matrix.shape[1]} {columns}.\n'
                   f'{explanations[key]}\n{legend}\n'
                   f'Attention row sums (head {self.head.get()}, rounded): {result["attention"].sum(-1).round(decimals=5).tolist()}\n'
                   'Each attention row totals 1 because it divides one unit of weight among allowed positions; this is not prediction accuracy.\n'
                   'Q/K/V, scores, attention and weighted_values show one head. Residual/FFN views use all heads, so changing Head leaves them unchanged.\n'
                   'Check passed: the explicit selected-block output matches Block.forward within numerical tolerance in evaluation mode (dropout off).\n'
                   'Open “Step-by-step walkthrough” for numerical examples and every View explained.')

    def render_4(self):
        ax = self.figure.add_subplot()
        if self.training:
            h = self.training.history
            ax.plot([r['step'] for r in h], [r['train'] for r in h], 'o-', color='#2165ad', label='Train (fixed windows)')
            ax.plot([r['step'] for r in h], [r['validation'] for r in h], 'o-', color='#c36b20', label='Validation (separate documents)')
            ax.legend()
            pair = self.training.sample_pair()
            self.write(f'Actual updates: {self.training.step}\nSame prompt {pair["prompt"]!r}; greedy decoding.\nBEFORE: {pair["before"]}\nAFTER: {pair["after"]}')
        else:
            self.write('Press Start to create a training run from the current shared model.\nA checkpoint loaded in the left menu is used as the starting point. Reset for a scratch experiment.\nSave produces before.pt, after.pt and report.json; these are comparison artifacts, not optimizer-resume checkpoints.')
        ax.set(xlabel='Optimizer update', ylabel='Mean next-token loss', title='Measured learning curve, CPU')

    def start_training(self):
        self.stop()
        try:
            if self.training is None:
                self.training = topic(4).TrainingRun(self.session, float(self.rate.get()))
            self.target_step = self.training.step + int(self.updates.get())
            self.train_tick()
        except Exception as exc:
            self.status.set(str(exc))

    def train_tick(self):
        self.timer = None
        try:
            self.training.advance()
            if self.training.step % 10 == 0 or self.training.step >= self.target_step:
                self.render()
            self.status.set(f'Training: {self.training.step}/{self.target_step} updates. Learning rate {self.training.learning_rate}; Stop pauses between updates.')
            if self.training.step < self.target_step:
                self.timer = self.after(1, self.train_tick)
        except Exception as exc:
            self.status.set(f'Training stopped: {exc}')

    def save_training(self):
        self.stop()
        if not self.training:
            return
        path = filedialog.asksaveasfilename(title='New directory name for run (not an existing file)')
        if path:
            try:
                self.status.set(f'Saved {self.training.save(path)}')
            except Exception as exc:
                self.status.set(str(exc))

    def reset_generation(self):
        self.generation = None
        self.render()

    def render_5(self):
        prompt = self.text.get()
        # Preview does not sample or advance the generation RNG.
        ids = self.session.ids(prompt) if self.generation is None else torch.tensor([self.generation.ids[-self.session.model.cfg.block_size:]])
        with torch.inference_mode():
            logits = self.session.model(ids)[0][0, -1]
            probabilities = (logits/float(self.temperature.get())).softmax(-1)
        ax = self.figure.add_subplot()
        ax.bar(range(len(probabilities)), probabilities.numpy(), color='#2874b9')
        ax.set_xticks(range(len(probabilities)), [repr(c) for c in self.session.vocabulary], rotation=90, fontsize=8)
        ax.set(title='Next-token probabilities for the current context', ylabel='Probability')
        text = prompt if self.generation is None else self.session.decode(self.generation.ids)
        self.write(f'Current text:\n{text}\n\nNext token uses the distribution above. No updates occur until you explicitly train in topic 004.')

    def generate_one(self):
        try:
            if self.generation is None:
                self.generation = topic(5).Generation(self.session, self.text.get())
            result = self.generation.advance(float(self.temperature.get()), self.greedy.get())
            self.render()
            self.status.set(f'Appended token ID {result["chosen"]}: {result["token"]!r}. Chart now predicts the following token.')
        except Exception as exc:
            self.status.set(str(exc))

    def render_6(self):
        axes = self.figure.subplots(1, 2)
        if self.adapter:
            layer = self.adapter.model.lm_head
            delta = (layer.b @ layer.a * layer.scale).detach()
            self.heatmap(axes[0], delta, 'Actual learned ΔW = (α/r) BA', ylabel='Vocabulary token')
            h = self.adapter.history
            axes[1].plot([v['step'] for v in h], [v['adaptation_loss'] for v in h], 'o-', color='#6c4298')
            pair = self.adapter.compare()
            self.write(f'Adapter shown: {self.adapter.domain}, rank {layer.a.shape[0]}, {self.adapter.step} updates\nBase: {self.adapter.base_label}\n'
                       f'Trainable A/B scalars: {sum(p.numel() for p in self.adapter.trainable)}\n'
                       f'BEFORE: {pair["base"]}\nWITH ADAPTER: {pair["adapter"]}\nCurve is adaptation training loss, not held-out domain performance. Change controls and click New adapter to start another.')
        else:
            self.write('Start a tiny adapter experiment. Train in 004 or load its checkpoint first for an adapted trained base.\n'
                       'With the initial shared model this only demonstrates low-rank optimization on random weights.\n'
                       'For real biology/email/finance instruction models, use train_pretrained.py and the topic guide. QLoRA is available there, not simulated here.')
        axes[1].set(xlabel='Adapter updates', ylabel='Training loss', title='Fixed adaptation window')

    def start_adapter(self):
        self.stop()
        try:
            self.adapter = topic(6).AdapterRun(self.session, self.domain.get(), int(self.rank.get()))
            self.adapter_tick()
        except Exception as exc:
            self.status.set(str(exc))

    def adapter_tick(self):
        self.timer = None
        try:
            self.adapter.advance()
            if self.adapter.step % 10 == 0:
                self.render()
            self.status.set(f'Adapter update {self.adapter.step}/60. Shared base stays unchanged.')
            if self.adapter.step < 60:
                self.timer = self.after(1, self.adapter_tick)
            else:
                self.adapter.assert_base_unchanged()
        except Exception as exc:
            self.status.set(f'Adapter stopped: {exc}')

    def save_adapter(self):
        self.stop()
        if not self.adapter:
            return
        path = filedialog.asksaveasfilename(title='New directory for tiny adapter artifacts')
        if path:
            try:
                self.status.set(f'Saved {self.adapter.save(path)}')
            except Exception as exc:
                self.status.set(str(exc))

    def render_7(self):
        result = topic(7).retrieve(self.text.get(), self.docs.get('1.0', 'end'))
        ax = self.figure.add_subplot()
        ax.barh(range(len(result['scores'])), result['scores'], color=['#258763' if i in result['selected'] else '#bfd0e4' for i in range(len(result['scores']))])
        ax.set_yticks(range(len(result['scores'])), [f'doc-{i+1}' for i in range(len(result['scores']))])
        ax.set(xlim=(0, 1), xlabel='TF-IDF cosine similarity', title='Green passages are supplied as evidence')
        self.write(result['prompt']+'\n\nNo LLM call is made in this view. answer_with_model.py performs actual generation with a staged model.')

    def render_8(self):
        predictions = self.predictions.get('1.0', 'end').strip().splitlines()
        metrics = topic(8).metrics(predictions)
        labels = ['positive', 'neutral', 'negative']
        matrix = [[metrics['confusion_matrix'][r][c] for c in labels+['INVALID']] for r in labels]
        ax = self.figure.add_subplot()
        ax.imshow(matrix, cmap='Blues', vmin=0)
        ax.set_xticks(range(4), labels+['INVALID'])
        ax.set_yticks(range(3), labels)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha='center', va='center', color='#12243b')
        ax.set(xlabel='Predicted label', ylabel='Reference label', title='Manual metric fixture · not a model benchmark')
        self.write(f'Fixed references: {topic(8).REFERENCES}\nAccuracy: {metrics["accuracy"]:.3f}\nMacro F1: {metrics["macro_f1"]:.3f}\nInvalid outputs: {metrics["invalid_output_rate"]:.3f}\n\nEdit the six predictions above. For actual generations click Compare two checkpoints; use before.pt and after.pt from 004 or merged.pt from 006.')

    def compare_checkpoints(self):
        left = filedialog.askopenfilename(title='Left checkpoint', filetypes=[('TinyGPT', '*.pt')])
        if not left:
            return
        right = filedialog.askopenfilename(title='Right checkpoint', filetypes=[('TinyGPT', '*.pt')])
        if not right:
            return
        try:
            a, b = Session(), Session()
            a.load(left)
            b.load(right)
            results = topic(8).compare(a, b, self.text.get())
            self.write('\n\n'.join(f'{r["model"]}\nPrompt: {r["prompt"]!r}\nOutput: {r["output"]}' for r in results)+'\n\nMeasured greedy outputs; the label chart above remains a separate manual fixture.')
        except Exception as exc:
            self.status.set(str(exc))


def launch(initial_topic=0):
    try:
        app = Lab(initial_topic)
    except tk.TclError as exc:
        raise SystemExit(f'A desktop display is required: {exc}\nRun this command in your local graphical terminal. On Roihu use the terminal demos instead.')
    app.mainloop()
