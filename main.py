import subprocess

while True:
    result = subprocess.run(['python', 'main_module.py'])
    if result.returncode == 3:
        break