"""Exercise every desktop view and controls. Requires a graphical desktop.

Optional --screenshot PATH captures only this lab's window, never the whole desktop.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
from llm_workshop.visual.app import Lab

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--screenshot', type=Path)
args = parser.parse_args()
app = Lab()
app.withdraw()
try:
    errors = []
    app.report_callback_exception = lambda typ, value, tb: errors.append(str(value))
    for number in range(9):
        app.select(number)
        app.update()
        assert app.status.get().startswith('Ready'), (number, app.status.get())
    app.select(1)
    app.text.set('the 🦉')
    app.refresh_input()
    assert 'Unsupported characters' in app.output.get('1.0', 'end')
    app.text.set('the cat sat.')
    app.refresh_input()
    app.select(3)
    for view in ['Q', 'K', 'V', 'masked_scores', 'attention', 'feed_forward', 'output']:
        app.view.set(view)
        app.render()
        assert app.status.get().startswith('Ready'), app.status.get()
    app.select(4)
    app.updates.set('10')
    app.start_training()
    deadline = time.monotonic() + 30
    while app.timer and time.monotonic() < deadline:
        app.update()
        time.sleep(.005)
    assert app.training.step == 10, app.status.get()
    app.select(5)
    app.generate_one()
    assert app.generation is not None
    app.select(6)
    app.start_adapter()
    deadline = time.monotonic() + 30
    while app.timer and time.monotonic() < deadline:
        app.update()
        time.sleep(.005)
    assert app.adapter.step == 60, app.status.get()
    app.select(7)
    app.docs.delete('1.0', 'end')
    app.docs.insert('1.0', 'The Neral instrument is stored in room M219.')
    app.render()
    assert 'M219' in app.output.get('1.0', 'end')
    assert not errors, errors
    if args.screenshot:
        from PIL import ImageGrab
        app.select(3)
        app.geometry('1260x920+40+40')
        app.deiconify()
        app.attributes('-topmost', True)
        app.update()
        time.sleep(.3)
        app.update()
        x, y = app.winfo_rootx(), app.winfo_rooty()
        ImageGrab.grab(bbox=(x, y, x+app.winfo_width(), y+app.winfo_height())).save(args.screenshot)
    print('Nine desktop views, editable text, head/views, training, generation, adapters and document edits passed.')
finally:
    app.close()
