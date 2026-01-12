from typing import List, Optional, Dict
from vidur.entities.full_request import FullRequest
from vidur.logger import init_logger

logger = init_logger(__name__)

class UnifiedRequestStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    
class UnifiedRequest:
    """
    UnifedRequest stands for a generic request that can be used for various types of requests, 
    such as single request, agent requests, etc.
    """
    def __init__(
        self,
        workflow_id: str, 
        arrive_at: float,
        workflow_config: List[Dict],  # 假设格式为: List[{"step": str, "input_str": str, "output_str": str}]
        max_token_for_request: int = 4096,
    ):
        self.workflow_id = workflow_id
        self.arrive_at = arrive_at
        self.workflow_config = workflow_config
        
        self.total_time = 0.0  # 记录在workflow内部已花费的时间
        self.completion_time = None # 记录workflow的完成时间
        self.request_counter = 0  # 用于生成唯一的 FullRequest ID
        
        self.step_names: List[str] = []  # different step names in workflow, distinguish from stage in vidur
        self.current_step_index: int = 0
        # active requests (submitted but not yet finished)
        self.active_requests: List[FullRequest] = []
        self.workflow_status = UnifiedRequestStatus.PENDING  

        self._initialize_step_names()
        self.total_steps = len(self.step_names)
        self.max_token_for_request = max_token_for_request   # for search data from profile data

        self.context_information = ""


    def _initialize_step_names(self):
        """
        Initialize the list of steps, maintaining the order.
        The simplest logic is used here: each configuration item is an independent step.
        """
        for step_config in self.workflow_config:
            # 去重
            if len(self.step_names) > 0 and self.step_names[-1] == step_config["step"]:
                continue
            self.step_names.append(step_config["step"])
        
    def is_finished(self) -> bool:
        """judge whether the workflow is finished"""
        # 当所有阶段都已处理，workflow完成
        return self.current_step_index >= self.total_steps

    def start(self, current_time: float):
        if self.workflow_status == UnifiedRequestStatus.PENDING:
            self.workflow_status = UnifiedRequestStatus.RUNNING 

    # 这个函数应该会在上一个full_request结束的event中调用或者初始时，用于发射下一个request
    # TODO(yinhan): 这里对于full request是否可以并行的逻辑实现得比较粗糙，后续可以考虑实现图结构
    def get_next_requests(self, current_time: float, content: str = "") -> List[FullRequest]:
        """
        Retrieve all FullRequests that need to be initiated in the current step.
        This is the core logic of the workflow: it only triggers after the previous step is completed (when active_requests is empty).
        """
        # 如果工作流已完成，或当前阶段仍在运行，则不产生新请求
        if self.is_finished() or len(self.active_requests) > 0:
            return []

        assert len(self.active_requests) == 0, "former level requests have not finished"
        
        # 检查是否所有阶段都已完成
        if self.current_step_index >= self.total_steps:
            self.workflow_status = UnifiedRequestStatus.COMPLETED
            self.completion_time = current_time
            logger.info(f"{self.workflow_id} completed, e2e latency is {(self.completion_time - self.arrive_at) * 1000} ms")
            return []

        new_requests: List[FullRequest] = []
        current_step_name = self.step_names[self.current_step_index]

        
        # 遍历，查找属于当前阶段的所有请求
        # 这里说明step的名字得是区分度的
        for step_config in self.workflow_config:
            if step_config["step"] == current_step_name:
                step_name_for_req = current_step_name

                next_request = FullRequest(
                    req_id=f"{self.workflow_id}_req_{self.request_counter}",
                    input_str=content + step_config["input_str"],
                    output_str=step_config["output_str"],   
                    arrived_at=current_time,
                    parent_unified_request=self,  # 链接回这个 UnifiedRequest 实例
                    max_tokens=self.max_token_for_request
                )

                if next_request.num_decode_tokens == 0 or next_request.num_prefill_tokens == 0:
                    continue

                # vidur 的上下文最大只有16384，小于数据集结果，因此剔除不合的request
                if next_request.num_prefill_tokens + next_request.num_decode_tokens > self.max_token_for_request:
                    continue

                assert next_request.num_decode_tokens > 0, f"Request {next_request.id}'s output str is empty."


                new_requests.append(next_request)
                self.request_counter += 1

        if len(new_requests) > 1:
            for req in new_requests:
                req.is_parallelizable = True

        self.active_requests.extend(new_requests)
        return new_requests

    def update_on_request_finish(self, finished_request: FullRequest, current_time: float):
        """
        The simulator should call this method when a FullRequest completes.
        This is used to update the workflow status and advance to the next stage.
        """
        if finished_request not in self.active_requests:
            return 

        self.active_requests.remove(finished_request)
        self.context_information += finished_request.input_str
        self.context_information += " "
        self.context_information += finished_request.output_str
        self.context_information += " "

        if not self.active_requests:
            self.current_step_index += 1
            
            # calculate workflow step duration
            # assume all request in a step are arrived at the same time
            step_start_time = finished_request.arrived_at
            step_duration = current_time - step_start_time
            self.total_time += step_duration
            
            if self.is_finished():
                self.workflow_status = UnifiedRequestStatus.COMPLETED
                self.completion_time = current_time

    def is_current_step_finished(self):
        if len(self.active_requests) == 0:
            return True
        else:
            return False