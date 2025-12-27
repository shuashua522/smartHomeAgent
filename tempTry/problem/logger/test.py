import logging
from logging.handlers import RotatingFileHandler
import os


def setup_logger(
        logger_name: str,  # 必传：logger名字
        log_file_path: str,  # 必传：日志文件存储路径（如 "./logs/app.log"）
        console_level: int = logging.INFO,  # 可选：控制台日志级别，默认INFO
        file_level: int = logging.DEBUG,  # 可选：文件日志级别，默认DEBUG
        max_bytes: int = 5 * 1024 * 1024,  # 可选：单个日志文件最大大小，默认5MB
        backup_count: int = 3,  # 可选：日志文件备份数量，默认3个
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S"
):
    """
    配置并返回一个自定义的日志器

    参数说明：
    - logger_name: 日志器名称（如 "user_service"、"order_service"）
    - log_file_path: 日志文件完整路径（如 "./logs/my_app.log"、"/var/log/app/debug.log"）
    - console_level: 控制台输出的日志级别（logging.DEBUG/INFO/WARNING/ERROR/CRITICAL）
    - file_level: 日志文件保存的日志级别
    - max_bytes: 单个日志文件最大字节数（超过则分割）
    - backup_count: 日志文件备份数量
    - log_format: 日志格式字符串
    - date_format: 时间格式字符串
    """
    # 1. 获取/创建指定名称的logger
    logger = logging.getLogger(logger_name)
    # 防止重复添加handler（多次调用setup_logger时避免日志重复输出）
    if logger.handlers:
        return logger

    # 2. 设置logger总级别（需≤处理器级别，否则会被过滤）
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False  # 避免日志向上传递到root logger导致重复输出

    # 3. 确保日志文件所在目录存在（如果路径包含多级目录，自动创建）
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 4. 定义日志格式
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 5. 创建控制台处理器（输出到屏幕）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # 6. 创建文件处理器（输出到指定路径的文件）
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"  # 确保中文日志不乱码
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # 7. 将处理器添加到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # 传入你关心的路径
    log_path = "/tmp/order.log"
    # 获取该路径在当前系统的实际绝对路径
    real_path = os.path.abspath(log_path)
    print(f"日志文件的实际存储路径：{real_path}")
    # 示例1：创建名为 "user_module" 的logger，日志文件保存到 ./logs/user.log
    user_logger = setup_logger(
        logger_name="user_module",
        log_file_path="./logs/user.log"
    )
    user_logger.info("用户模块初始化完成")
    user_logger.debug("用户模块调试信息：加载配置成功")

    # # 示例2：创建名为 "order_module" 的logger，日志文件保存到 /tmp/order.log
    # order_logger = setup_logger(
    #     logger_name="order_module",
    #     log_file_path="/tmp/order.log",
    #     console_level=logging.WARNING,  # 控制台只输出WARNING及以上
    #     max_bytes=10 * 1024 * 1024  # 单个文件最大10MB
    # )
    # order_logger.warning("订单模块警告：库存不足")
    # order_logger.error("订单模块错误：创建订单失败")