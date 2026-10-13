"""Interactive SSH bridge launched by Ghostty; password stays in process memory."""
import argparse
import json
import os
from pathlib import Path
import pty
import re
import select
import signal
import sys
import termios
import tty

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/ghostty-short-drift-20261012-v1'


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='ghostty-short-drift-20261012-v1')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9-]+', args.run):
        raise ValueError('Invalid bridge run name')
    OUT = ROOT / 'artifacts' / args.run
    OUT.mkdir(parents=True, exist_ok=False)
    queue = OUT / 'commands'
    queue.mkdir()
    pid, master = pty.fork()
    if pid == 0:
        os.execv('/usr/bin/ssh', ['ssh', '-tt', '-o', 'ConnectTimeout=15', '-o',
                                 'ServerAliveInterval=15', '-p', '2020',
                                 '19905334880@xn-b.suanjiayun.com'])
    status = {'state': 'connecting', 'ssh_pid': pid, 'commands_sent': [], 'password_saved': False}
    def save():
        (OUT / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    save()
    pending = ''
    selected = False
    stdin_fd = sys.stdin.fileno()
    original_termios = termios.tcgetattr(stdin_fd)
    tty.setraw(stdin_fd)
    try:
        with (OUT / 'session.log').open('ab', buffering=0) as log:
            while True:
                ready, _, _ = select.select([master, stdin_fd], [], [], .3)
                if stdin_fd in ready:
                    typed = os.read(stdin_fd, 4096)
                    if not typed:
                        break
                    os.write(master, typed)
                if master in ready:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                    log.write(data)
                    pending = (pending + data.decode('utf-8', errors='replace'))[-8192:]
                    clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', pending)
                    if clean.rstrip().lower().endswith('password:'):
                        status['state'] = 'waiting_password'
                        save()
                    if not selected and '输入序号>' in clean:
                        os.write(master, b'0\r')
                        selected = True
                        status['state'] = 'instance_selected'
                        save()
                    if re.search(r'root@[^\r\n]+[#] ', clean):
                        status['state'] = 'shell_ready'
                        save()
                if status['state'] == 'shell_ready':
                    for path in sorted(queue.glob('*.cmd')):
                        if path.name in status['commands_sent']:
                            continue
                        command = path.read_bytes()
                        os.write(master, command if command.endswith(b'\n') else command + b'\n')
                        status['commands_sent'].append(path.name)
                        save()
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, original_termios)
        status['state'] = 'connection_closed'
        save()
        os.close(master)
        try:
            os.kill(pid, signal.SIGHUP)
        except ProcessLookupError:
            pass


if __name__ == '__main__':
    main()
