import datetime

class FileLogger:
    def __init__(self):
        self.log_file = open('log.log', 'a')
    def log(self, p_type, value):
        now = datetime.datetime.now()
        self.log_file.write(str(now.strftime("%Y-%m-%d %H:%M:%S")) + " : " + p_type + " = " + value + '\n')
    def __del__(self):
        self.log_file.write("\n\n")