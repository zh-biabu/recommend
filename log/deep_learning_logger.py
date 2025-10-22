import logging
import os
import datetime
from logging.handlers import TimedRotatingFileHandler
import json


class DeepLearningLogger(logging.Logger):
    """
    深度学习专用日志对象，继承自logging.Logger
    支持按天分割日志文件，专门记录训练过程
    """
    
    def __init__(self, name="DeepLearningLogger", level=logging.INFO):
        super().__init__(name, level)
        
        # 创建train_log目录
        self.train_log_dir = "./log/train_log"
        # if not os.path.exists(self.train_log_dir):
        #     os.makedirs(self.train_log_dir)
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 设置控制台处理器
        self._setup_console_handler()
        
        # 设置文件处理器（按天分割）
        self._setup_file_handler()
    
    def _setup_console_handler(self):
        """设置控制台输出处理器"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        self.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """设置文件处理器，按天分割日志"""
        if not os.path.exists(self.train_log_dir):
            os.makedirs(self.train_log_dir, exist_ok=True)  # 加 exist_ok 避免多进程冲突

        log_file = os.path.join(self.train_log_dir, "training.log")
        
        # 使用TimedRotatingFileHandler按天分割日志
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',  # 每天午夜分割
            interval=1,       # 间隔1天
            backupCount=30,   # 保留30天的日志
            encoding='utf-8',
            delay=True
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.formatter)
        
        # 设置日志文件名后缀格式
        file_handler.suffix = "%Y-%m-%d"
        
        self.addHandler(file_handler)
    
    def log_epoch_start(self, epoch, total_epochs):
        """记录训练轮次开始"""
        self.info(f"=" * 60)
        self.info(f"开始第 {epoch}/{total_epochs} 轮训练")
        self.info(f"=" * 60)
    
    def log_epoch_end(self, epoch, metrics):
        """记录训练轮次结束及指标"""
        self.info(f"第 {epoch} 轮训练完成")
        for metric_name, value in metrics.items():
            self.info(f"{metric_name}: {value:.6f}")
        self.info("-" * 60)
    
    def log_batch_progress(self, batch_idx, total_batches, loss, lr=None):
        """记录批次训练进度"""
        progress = (batch_idx + 1) / total_batches * 100
        msg = f"批次 [{batch_idx+1}/{total_batches}] ({progress:.1f}%) - Loss: {loss:.6f}"
        if lr is not None:
            msg += f" - LR: {lr:.8f}"
        self.debug(msg)
    
    def log_validation_results(self, metrics):
        """记录验证结果"""
        self.info("验证结果:")
        for metric_name, value in metrics.items():
            self.info(f"  {metric_name}: {value:.6f}")
    
    def log_model_info(self, model_name, total_params, trainable_params):
        """记录模型信息"""
        self.info(f"模型: {model_name}")
        self.info(f"总参数量: {total_params:,}")
        self.info(f"可训练参数量: {trainable_params:,}")
    
    def log_hyperparameters(self, hyperparams):
        """记录超参数"""
        self.info("超参数配置:")
        for param_name, value in hyperparams.items():
            self.info(f"  {param_name}: {value}")
    
    def log_training_config(self, config):
        """记录训练配置"""
        self.info("训练配置:")
        self.info(f"  批次大小: {config.get('batch_size', 'N/A')}")
        self.info(f"  学习率: {config.get('learning_rate', 'N/A')}")
        self.info(f"  优化器: {config.get('optimizer', 'N/A')}")
        self.info(f"  损失函数: {config.get('loss_function', 'N/A')}")
        self.info(f"  设备: {config.get('device', 'N/A')}")
    
    def log_checkpoint_save(self, epoch, checkpoint_path):
        """记录检查点保存"""
        self.info(f"检查点已保存: Epoch {epoch} -> {checkpoint_path}")
    
    def log_early_stopping(self, epoch, best_metric):
        """记录早停信息"""
        self.warning(f"早停触发 - 第 {epoch} 轮，最佳指标: {best_metric:.6f}")
    
    def log_training_complete(self, total_time, best_metrics):
        """记录训练完成信息"""
        self.info("=" * 60)
        self.info("训练完成!")
        self.info(f"总训练时间: {total_time}")
        self.info("最佳指标:")
        for metric_name, value in best_metrics.items():
            self.info(f"  {metric_name}: {value:.6f}")
        self.info("=" * 60)
    
    def log_error(self, error_msg, exception=None):
        """记录错误信息"""
        self.error(f"训练错误: {error_msg}")
        if exception:
            self.exception(exception)
    
    def log_metrics_to_json(self, epoch, metrics, json_file="training_metrics.json"):
        """将指标保存到JSON文件"""
        json_path = os.path.join(self.train_log_dir, json_file)
        
        # 读取现有数据
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        # 添加新数据
        metrics_entry = {
            'epoch': epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            **metrics
        }
        data.append(metrics_entry)
        
        # 保存数据
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.debug(f"指标已保存到 {json_path}")


def get_logger(name="DeepLearningLogger", level=logging.INFO):
    """
    获取深度学习日志对象的工厂函数
    
    Args:
        name: 日志器名称
        level: 日志级别
    
    Returns:
        DeepLearningLogger实例
    """
    # 设置自定义Logger类
    logging.setLoggerClass(DeepLearningLogger)
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if not logger.handlers:
        logger.__init__(name, level)
    
    return logger


# 使用示例
if __name__ == "__main__":
    # 创建日志对象
    logger = get_logger()
    
    # 示例：记录训练过程
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'device': 'cuda'
    }
    
    hyperparams = {
        'epochs': 100,
        'weight_decay': 1e-4,
        'dropout': 0.5
    }
    
    # 记录训练配置
    logger.log_training_config(config)
    logger.log_hyperparameters(hyperparams)
    logger.log_model_info("ResNet50", 25557032, 25557032)
    
    # 模拟训练过程
    for epoch in range(1, 4):
        logger.log_epoch_start(epoch, 3)
        
        # 模拟批次训练
        for batch in range(100):
            if batch % 20 == 0:
                logger.log_batch_progress(batch, 100, 0.5 - epoch * 0.1, 0.001)
        
        # 记录轮次结束
        metrics = {
            'train_loss': 0.5 - epoch * 0.1,
            'train_acc': 0.7 + epoch * 0.05,
            'val_loss': 0.6 - epoch * 0.08,
            'val_acc': 0.65 + epoch * 0.06
        }
        
        logger.log_epoch_end(epoch, metrics)
        logger.log_metrics_to_json(epoch, metrics)
        
        if epoch == 2:
            logger.log_checkpoint_save(epoch, f"model_epoch_{epoch}.pth")
    
    # 记录训练完成
    best_metrics = {'best_val_acc': 0.83, 'best_val_loss': 0.36}
    logger.log_training_complete("2小时30分钟", best_metrics)
