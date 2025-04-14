import logging


def get_logger(log_file):
    logger = logging.getLogger("logger")    # 创建一个日志器
    logger.setLevel(logging.INFO)           # 设置日志输出的最低等级，低于当前等级则会被忽略
    if not logger.handlers:                 # 判断当前日志对象中是否有处理器，如果没有，则添加处理器
        sh = logging.StreamHandler()        # 创建处理器：sh为控制台处理器，fh为文件处理器，log_file为日志存放的文件夹
        fh = logging.FileHandler(log_file, encoding="UTF-8")
        formator = logging.Formatter(
            fmt="%(asctime)s %(filename)s %(levelname)s %(message)s",
            datefmt="%Y/%m/%d %X",
        )                                   # 创建格式器,并将sh，fh设置对应的格式
        sh.setFormatter(formator)
        fh.setFormatter(formator)
        logger.addHandler(sh)               # 将处理器，添加至日志器中
        logger.addHandler(fh)
    return logger