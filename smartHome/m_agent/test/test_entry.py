

token=HOMEASSITANT_AUTHORIZATION_TOKEN
server=HOMEASSITANT_SERVER
active_project_env=ACTIVE_PROJECT_ENV

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 定位到所在目录
current_dir = os.path.dirname(current_file_path)
testing_logs_dir = os.path.join(current_dir, "testing_logs")

@dataclass
class AgentResourceStats:
    """Agent一次运行的资源统计结果"""
    run_time: float  # 总运行时间（秒）
    memory_peak: float  # 内存峰值（MB）
    memory_avg: float  # 内存平均值（MB）
    cpu_peak: float  # CPU使用率峰值（%）
    cpu_avg: float  # CPU使用率平均值（%）
class AgentResourceMonitor:
    def __init__(self, sample_interval: float = 0.1):
        """
        初始化监控器
        :param sample_interval: 采样间隔（秒），越小越精确但开销略高
        """
        self.sample_interval = sample_interval
        self.pid = psutil.Process()  # 当前进程
        self._running = False
        self._memory_samples: List[float] = []  # 内存采样（MB）
        self._cpu_samples: List[float] = []  # CPU采样（%）
        self._start_time: float = 0.0

    def _sample_loop(self):
        """后台采样循环"""
        self._running = True
        # 初始化CPU使用率计算（首次调用返回0，需先触发）
        self.pid.cpu_percent(interval=0.01)
        while self._running:
            # 记录内存（RSS：物理内存，转换为MB）
            mem_rss = self.pid.memory_info().rss / (1024 **2)
            self._memory_samples.append(mem_rss)
            # 记录CPU使用率（基于采样间隔的平均值）
            cpu_percent = self.pid.cpu_percent(interval=self.sample_interval)
            self._cpu_samples.append(cpu_percent)

    def start(self):
        """开始监控"""
        self._start_time = time.time()
        self._memory_samples.clear()
        self._cpu_samples.clear()
        # 启动采样线程（守护线程，随主程序退出）
        self._sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sample_thread.start()

    def stop(self) -> AgentResourceStats:
        """停止监控并返回统计结果"""
        self._running = False
        self._sample_thread.join()  # 等待采样线程结束
        run_time = time.time() - self._start_time

        # 计算内存指标（过滤空采样的极端情况）
        if self._memory_samples:
            memory_peak = max(self._memory_samples)
            memory_avg = sum(self._memory_samples) / len(self._memory_samples)
        else:
            memory_peak = memory_avg = 0.0

        # 计算CPU指标
        if self._cpu_samples:
            cpu_peak = max(self._cpu_samples)
            cpu_avg = sum(self._cpu_samples) / len(self._cpu_samples)
        else:
            cpu_peak = cpu_avg = 0.0

        return AgentResourceStats(
            run_time=run_time,
            memory_peak=memory_peak,
            memory_avg=memory_avg,
            cpu_peak=cpu_peak,
            cpu_avg=cpu_avg
        )

def init_test_global_config():
    """测试前都要初始化测试环境"""
    global_config.TOKEN_TRACKING_CALLBACK=TokenTrackingCallback()


def process_testcases(agent_name,dir_path=testing_logs_dir,testNums=1):
    """
    遍历测试用例
    agent_name取值：[singleAgent,sashaAgent,ourAgent]
    """
    if agent_name not in ["singleAgent","sashaAgent","sageAgent","privacyAgent"]:
        raise ValueError("无效的arg：agent_name")

    dir_path = os.path.join(testing_logs_dir, global_config.MODEL,str(testNums), agent_name)
    # 确保目标目录存在（不存在则创建）
    os.makedirs(dir_path, exist_ok=True)

    init_test_global_config()
    # 遍历测试用例
    from test_cases import init_devices
    for index, func in enumerate(init_devices.registered_functions):
        # if(index+1<=36):
        #     continue
        # if(index+1<=23):
        #     continue
        question=func()

        # 处理文件名：移除特殊字符，确保文件名合法 ; 保留中文、字母、数字和下划线，其他字符替换为下划线
        cleaned_name = re.sub(r'[^\w\u4e00-\u9fa5]', '', question)
        # 生成文件名：序号_清洗后的内容（如"0_将整个房子变暗"、"1_网络状况"）
        filename = f"{index+1}_{cleaned_name}.log"
        # 生成文件完整路径
        log_file = os.path.join(dir_path, filename)
        logger=get_context_logger(log_file=log_file, name=f"{agent_name}_{index}")

        # 配置全局信息
        global_config.GLOBAL_AGENT_DETAILED_LOGGER=logger
        # todo 真正测试前开启
        os.environ["LANGSMITH_PROJECT"] = agent_name
        global_config.LANGSMITH_TAG_NAME=f"{agent_name}_{global_config.MODEL}_{testNums}"

        # 调用agent
        # logger.info(f"test-{global_config.PROVIDER}")
        monitor = AgentResourceMonitor(sample_interval=0.1)
        monitor.start()
        try:
            if agent_name=="singleAgent":
                SingleAgent(logger=logger).run_agent(question)
            elif agent_name=="sashaAgent":
                run_sashaAgent(question)
            elif agent_name=="sageAgent":
                SageAgent().run_agent(question)
            elif agent_name=="ourAgent" or agent_name=="privacyAgent":
                # SmartHomeAgent().run_agent(question)
                file_path = r"F:\PyCharm\langchain_test\agent_project\agentcore\smart_home_agent\generate_conditional_code\condtional_code.py"
                # 以写入模式（'w'）打开文件，该模式会自动清空文件原有内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    pass
                privacy_home_agent(question)
        except Exception as e:
            # 1. 获取完整的异常信息（类型、消息、堆栈跟踪）
            # traceback.format_exc() 会返回包含堆栈的字符串，便于调试
            exception_detail = traceback.format_exc()

            # 2. 将异常原样打印到日志（使用error级别，突出异常）
            logger.error(
                f"执行Agent [{agent_name}] 时发生异常：\n"
                f"异常类型：{type(e).__name__}\n"
                f"异常消息：{str(e)}\n"
                f"完整堆栈跟踪：\n{exception_detail}"
            )

            # （可选）若需要向上层传递异常，可取消注释下面一行（根据业务需求决定）
            # raise
        stats = monitor.stop()
        callback = global_config.TOKEN_TRACKING_CALLBACK;
        logger.info(callback.get_agent_total_usage())

        logger.info("\nAgent运行一次的资源统计结果：")
        logger.info(f"总运行时间：{stats.run_time:.2f}秒")
        logger.info(f"内存峰值：{stats.memory_peak:.2f}MB")
        logger.info(f"内存平均值：{stats.memory_avg:.2f}MB")
        logger.info(f"CPU使用率峰值：{stats.cpu_peak:.2f}%")
        logger.info(f"CPU使用率平均值：{stats.cpu_avg:.2f}%")

def main(agent_name,testNums):
    process_testcases(agent_name=agent_name,testNums=testNums)


if __name__=="__main__":
    # main("singleAgent",1)
    # main("sashaAgent",1)
    # main("sageAgent",1)
    main("privacyAgent",4)
    pass