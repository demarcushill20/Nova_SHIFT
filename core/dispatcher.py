"""
Core component responsible for dispatching tasks to available agents in the swarm.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Set
import logging
from collections import deque

# Assuming SharedMemoryInterface is importable
from .shared_memory_interface import SharedMemoryInterface
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder for actual Agent representation
AgentID = str
SubtaskID = str
Subtask = Dict[str, Any] # Define type alias for subtask dictionary

# Constants for heartbeat monitoring
HEARTBEAT_CHECK_INTERVAL = 15  # seconds
HEARTBEAT_TIMEOUT = 60  # seconds (agent considered failed if no heartbeat for this long)

class SwarmDispatcher:
    """Manages a pool of agents, assigns subtasks, and monitors agent health.

    Handles task distribution using round-robin, tracks agent availability,
    and manages agent failures through heartbeat monitoring.

    Attributes:
        _agents: Dictionary storing agent information (status, instance).
        _agent_ids_round_robin: List of agent IDs for round-robin assignment.
        _current_agent_index: Index for the next agent in round-robin.
        _task_assignments: Mapping of assigned subtask IDs to agent IDs.
        _pending_tasks: Queue of subtasks waiting for assignment.
        _failed_agents: Set of agent IDs marked as failed.
        _shared_memory: Instance of SharedMemoryInterface.
        _heartbeat_monitor_task: Asyncio task for the heartbeat monitor.
        _running: Flag indicating if the heartbeat monitor is active.
    """

    def __init__(self, shared_memory: SharedMemoryInterface):
        """Initializes the SwarmDispatcher.

        Args:
            shared_memory (SharedMemoryInterface): An instance of
                SharedMemoryInterface used for agent heartbeat checks and potentially
                other coordination tasks.
        """
        self._agents: Dict[AgentID, Dict[str, Any]] = {}  # agent_id -> {"status": "available" | "busy" | "failed", "instance": AgentInstance}
        self._agent_ids_round_robin: List[AgentID] = []
        self._current_agent_index = 0
        self._task_assignments: Dict[SubtaskID, AgentID] = {} # subtask_id -> agent_id
        self._pending_tasks: deque[Subtask] = deque() # Queue for tasks waiting for an agent
        self._failed_agents: Set[AgentID] = set() # Keep track of failed agents
        self._shared_memory = shared_memory
        self._heartbeat_monitor_task: Optional[asyncio.Task] = None
        self._running = False
        logger.info("SwarmDispatcher initialized.")

    def register_agent(self, agent_id: AgentID, agent_instance: Any):
        """Registers a new agent or re-registers an existing one.

        Adds the agent to the pool, marks it as available, and includes it
        in the round-robin assignment list if not already present. If the agent
        was previously marked as failed, this clears the failed status.

        Args:
            agent_id (AgentID): The unique identifier for the agent.
            agent_instance (Any): The actual agent object or a reference to it,
                expected to have an `execute_task` async method.
        """
        if agent_id not in self._agents:
            self._agents[agent_id] = {"status": "available", "instance": agent_instance}
            # Ensure agent is not marked as failed if re-registering
            self._failed_agents.discard(agent_id)
            if agent_id not in self._agent_ids_round_robin:
                self._agent_ids_round_robin.append(agent_id)
            logger.info(f"Agent registered or re-registered: {agent_id}")
        else:
            logger.warning(f"Agent {agent_id} already registered.")

    async def unregister_agent(self, agent_id: AgentID): # Made async
        """Unregisters an agent, marking it as failed and handling its tasks.

        Marks the agent as failed, attempts to requeue its assigned tasks,
        and removes it from the active agent pool and round-robin list.

        Args:
            agent_id (AgentID): The unique identifier for the agent to remove.
        """
        if agent_id in self._agents:
            # Mark as failed instead of deleting immediately to handle assigned tasks
            await self._handle_agent_failure(agent_id, reason="unregistered")
            # Clean up internal lists after handling failure
            if agent_id in self._agent_ids_round_robin:
                 self._agent_ids_round_robin.remove(agent_id)
                 if self._current_agent_index >= len(self._agent_ids_round_robin):
                     self._current_agent_index = 0
            if agent_id in self._agents:
                 del self._agents[agent_id] # Now remove from main dict
            logger.info(f"Agent unregistered and removed: {agent_id}")
        else:
            logger.warning(f"Agent {agent_id} not found for unregistration.")

    def _get_next_available_agent(self) -> Optional[AgentID]:
        """Finds the next available and non-failed agent using round-robin.

        Iterates through the registered agents starting from the last assigned index,
        skipping failed agents, until an 'available' agent is found.

        Returns:
            Optional[AgentID]: The ID of the next available agent, or None if no
            suitable agent is found.
        """
        if not self._agent_ids_round_robin:
            return None

        start_index = self._current_agent_index
        num_agents = len(self._agent_ids_round_robin)

        for i in range(num_agents):
            agent_index = (start_index + i) % num_agents
            agent_id = self._agent_ids_round_robin[agent_index]
            # Skip failed agents
            if agent_id in self._failed_agents:
                continue
            # Check status
            if self._agents[agent_id]["status"] == "available":
                self._current_agent_index = (agent_index + 1) % num_agents
                return agent_id

        logger.warning("No available agents found.")
        return None

    async def dispatch_subtasks(self, new_subtasks: List[Subtask]) -> Dict[SubtaskID, AgentID]:
        """Assigns pending and new subtasks to available agents.

        Adds new subtasks to the pending queue and attempts to assign tasks
        from the queue to available agents using a round-robin strategy.
        If an agent is found, the task is assigned, the agent is marked busy,
        and the agent's `execute_task` method is launched asynchronously.
        If no agent is available, remaining tasks stay in the pending queue.

        Args:
            new_subtasks (List[Subtask]): A list of new subtask dictionaries to add
                to the pending queue before attempting dispatch. Each dictionary
                should contain at least 'subtask_id'.

        Returns:
            Dict[SubtaskID, AgentID]: A dictionary mapping the subtask IDs that were
            successfully assigned in this call to the agent ID they were assigned to.
        """
        assignments: Dict[SubtaskID, AgentID] = {}
        if not self._agents:
            logger.error("Cannot dispatch tasks: No agents registered.")
            return assignments

        # Add new tasks to the pending queue
        for task in new_subtasks:
            self._pending_tasks.append(task)

        logger.info(f"Dispatch cycle starting. Pending tasks: {len(self._pending_tasks)}")

        processed_assignments: Dict[SubtaskID, AgentID] = {}
        tasks_to_keep_pending: deque[Subtask] = deque()

        while self._pending_tasks:
            subtask = self._pending_tasks.popleft()
            subtask_id = subtask.get("subtask_id")

            if not subtask_id:
                logger.warning(f"Pending subtask missing 'subtask_id': {subtask}. Discarding.")
                continue

            # Check if task was already assigned (e.g., from a previous failed attempt)
            if subtask_id in self._task_assignments:
                logger.warning(f"Task {subtask_id} is already assigned to {self._task_assignments[subtask_id]}. Skipping redundant dispatch.")
                continue

            # Basic round-robin assignment
            assigned_agent_id = self._get_next_available_agent()

            if assigned_agent_id:
                processed_assignments[subtask_id] = assigned_agent_id
                self._agents[assigned_agent_id]["status"] = "busy" # Mark agent as busy
                self._task_assignments[subtask_id] = assigned_agent_id
                logger.info(f"Assigned subtask {subtask_id} to agent {assigned_agent_id}")

                # --- Trigger agent execution (Phase 2.T4) ---
                # Get the actual agent instance
                agent_instance = self._agents[assigned_agent_id].get("instance")
                if agent_instance and hasattr(agent_instance, 'execute_task') and asyncio.iscoroutinefunction(agent_instance.execute_task):
                    # Launch the agent's task asynchronously
                    asyncio.create_task(agent_instance.execute_task(subtask))
                    logger.debug(f"Launched task {subtask_id} execution for agent {assigned_agent_id}")
                else:
                    logger.error(f"Agent {assigned_agent_id} instance not found or has no async execute_task method.")
                # ---------------------------------------------

            else:
                # No available agent found, put task back in queue for next cycle
                logger.warning(f"No available agent for subtask {subtask_id}. Re-queuing.")
                tasks_to_keep_pending.append(subtask)
                # Optimization: If we tried to assign and failed, stop trying this cycle
                # as all agents are likely busy or failed.
                break # Stop processing the queue for this cycle

        # Add any tasks that couldn't be assigned back to the main pending queue
        self._pending_tasks.extendleft(reversed(tasks_to_keep_pending)) # Add back to front

        logger.info(f"Dispatch cycle finished. Assigned: {len(processed_assignments)}, Remaining pending: {len(self._pending_tasks)}")
        return processed_assignments

    def update_task_status(self, subtask_id: SubtaskID, status: str, result: Any = None):
        """Updates the status of a task and the assigned agent.

        Called by agents (or potentially other components) when a task finishes
        or fails. If the task is 'completed' or 'failed', the assigned agent
        (if known and not already failed) is marked as 'available', and the
        task assignment is removed. Triggers a dispatch cycle if tasks are pending.

        Args:
            subtask_id (SubtaskID): The ID of the subtask being updated.
            status (str): The new status (e.g., "completed", "failed").
            result (Any, optional): The result of the task, if completed successfully,
                or an error message/object if failed. Defaults to None.
        """
        logger.info(f"Received status update for subtask {subtask_id}: {status}")
        if subtask_id in self._task_assignments:
            agent_id = self._task_assignments[subtask_id]
            if status in ["completed", "failed"]:
                # Check if agent is known and not already marked failed
                if agent_id in self._agents and agent_id not in self._failed_agents:
                    self._agents[agent_id]["status"] = "available" # Mark agent as available again
                    logger.info(f"Agent {agent_id} finished task {subtask_id} and is now available.")
                elif agent_id in self._failed_agents:
                     logger.warning(f"Agent {agent_id} reported status for task {subtask_id} but was already marked as failed.")
                else:
                     logger.warning(f"Agent {agent_id} reported status for task {subtask_id}, but is not registered.")

                # Remove completed/failed task assignment regardless of agent status
                del self._task_assignments[subtask_id]

            # Attempt to dispatch pending tasks now that an agent might be free
            if self._pending_tasks:
                 logger.info("Triggering dispatch for pending tasks after agent became available.")
                 # Schedule dispatch_subtasks without adding new tasks
                 asyncio.create_task(self.dispatch_subtasks([]))
        else:
            logger.warning(f"Received status update for unknown or unassigned subtask {subtask_id}")


    def get_agent_status(self) -> Dict[AgentID, str]:
        """Returns the current status of all registered agents.

        Provides a snapshot of the known status ('available', 'busy', 'failed')
        for each agent currently managed by the dispatcher.

        Returns:
            Dict[AgentID, str]: A dictionary mapping agent IDs to their status string.
        """
        return {agent_id: data["status"] for agent_id, data in self._agents.items()}

    async def _handle_agent_failure(self, agent_id: AgentID, reason: str = "heartbeat timeout"):
        """Handles the failure of an agent, marking it and requeueing its tasks.

        Marks the agent as 'failed', adds it to the `_failed_agents` set,
        and attempts to find and requeue any tasks currently assigned to it.

        Args:
            agent_id (AgentID): The ID of the agent that failed.
            reason (str, optional): The reason for the failure.
                Defaults to "heartbeat timeout".
        """
        if agent_id in self._failed_agents:
            return # Already handled

        logger.error(f"Agent {agent_id} failure detected! Reason: {reason}")
        self._failed_agents.add(agent_id)
        if agent_id in self._agents:
            self._agents[agent_id]["status"] = "failed" # Mark status

        # Find tasks assigned to the failed agent and requeue them
        tasks_to_requeue: List[SubtaskID] = [
            task_id for task_id, assigned_agent in self._task_assignments.items()
            if assigned_agent == agent_id
        ]

        if tasks_to_requeue:
            logger.warning(f"Requeuing tasks assigned to failed agent {agent_id}: {tasks_to_requeue}")
            # Need the original subtask definitions to requeue
            # This requires storing the subtask definition somewhere accessible,
            # e.g., in Shared Memory when first dispatched or keeping a local cache.
            # For now, we'll just remove the assignment. A more robust implementation
            # needs to retrieve the full subtask definition.
            for task_id in tasks_to_requeue:
                # TODO: Retrieve full subtask definition to add to self._pending_tasks
                logger.warning(f"Task {task_id} needs full definition retrieval for proper requeueing.")
                # Simple removal for now:
                if task_id in self._task_assignments:
                    del self._task_assignments[task_id]
                # Ideally:
                # subtask_def = await self._shared_memory.read(f"task:{task_id}:definition")
                # if subtask_def:
                #     self._pending_tasks.append(subtask_def)
                # else:
                #     logger.error(f"Could not retrieve definition for task {task_id} to requeue.")

            # Trigger dispatch for potentially requeued tasks
            if self._pending_tasks:
                 logger.info("Triggering dispatch for potentially requeued tasks.")
                 asyncio.create_task(self.dispatch_subtasks([]))
        else:
             logger.info(f"Failed agent {agent_id} had no active tasks assigned.")

    async def _check_agent_heartbeats(self):
        """Periodically checks agent heartbeats stored in Shared Memory (Redis).

        Runs as a background asyncio task. Iterates through registered, non-failed
        agents, reads their last heartbeat timestamp from Shared Memory, and compares
        it to the current time. If the time difference exceeds `HEARTBEAT_TIMEOUT`,
        the agent is marked as failed using `_handle_agent_failure`.
        """
        logger.info("Heartbeat monitor started.")
        while self._running:
            await asyncio.sleep(HEARTBEAT_CHECK_INTERVAL)
            logger.debug("Checking agent heartbeats...")
            current_time = time.time()
            # Check only registered and not already failed agents
            agents_to_check = list(self._agents.keys() - self._failed_agents)

            for agent_id in agents_to_check:
                heartbeat_key = f"agent:{agent_id}:heartbeat"
                try:
                    last_heartbeat_str = await self._shared_memory.read(heartbeat_key)
                    if last_heartbeat_str is None:
                        # Agent might have just started or never sent a heartbeat
                        # Consider a grace period or check if it's busy
                        if self._agents[agent_id]["status"] == "busy":
                             logger.warning(f"Agent {agent_id} is busy but has no heartbeat key. Potential issue.")
                        continue # Skip check if no heartbeat recorded yet

                    last_heartbeat = float(last_heartbeat_str)
                    if current_time - last_heartbeat > HEARTBEAT_TIMEOUT:
                        logger.warning(f"Agent {agent_id} heartbeat timeout detected (last seen {current_time - last_heartbeat:.1f}s ago).")
                        await self._handle_agent_failure(agent_id)

                except (TypeError, ValueError):
                     logger.error(f"Invalid heartbeat timestamp format for agent {agent_id} in Redis key '{heartbeat_key}'.")
                except Exception as e:
                     logger.error(f"Error checking heartbeat for agent {agent_id}: {e}", exc_info=True)
            logger.debug("Heartbeat check finished.")
        logger.info("Heartbeat monitor stopped.")

    async def start_monitoring(self):
        """Starts the background heartbeat monitor task if not already running."""
        if self._heartbeat_monitor_task is None or self._heartbeat_monitor_task.done():
            self._running = True
            self._heartbeat_monitor_task = asyncio.create_task(self._check_agent_heartbeats())
            logger.info("Scheduled heartbeat monitor task.")
        else:
             logger.warning("Heartbeat monitor task already running.")

    async def stop_monitoring(self):
        """Stops the background heartbeat monitor task gracefully."""
        self._running = False
        if self._heartbeat_monitor_task and not self._heartbeat_monitor_task.done():
            self._heartbeat_monitor_task.cancel()
            try:
                await self._heartbeat_monitor_task
            except asyncio.CancelledError:
                logger.info("Heartbeat monitor task cancelled successfully.")
            except Exception as e:
                 logger.error(f"Error during heartbeat monitor task cancellation: {e}", exc_info=True)
        self._heartbeat_monitor_task = None

# Example Usage (for testing purposes)
async def main():
    # Requires SharedMemoryInterface instance
    mock_shared_memory = MagicMock(spec=SharedMemoryInterface)
    mock_shared_memory.read = AsyncMock(return_value=None)
    mock_shared_memory.write = AsyncMock()
    dispatcher = SwarmDispatcher(shared_memory=mock_shared_memory)
    # Start monitoring (in real app, this would be managed by the main application lifecycle)
    # await dispatcher.start_monitoring()

    # Simulate registering agents
    class MockAgent:
        def __init__(self, agent_id):
            self.id = agent_id
        async def execute_task(self, task):
            logger.info(f"Agent {self.id} starting task {task['subtask_id']}")
            await asyncio.sleep(1) # Simulate work
            logger.info(f"Agent {self.id} finished task {task['subtask_id']}")
            # In real scenario, agent would call dispatcher.update_task_status
            dispatcher.update_task_status(task['subtask_id'], "completed", f"Result from {self.id}")

    agent1 = MockAgent("agent_001")
    agent2 = MockAgent("agent_002")
    dispatcher.register_agent(agent1.id, agent1)
    dispatcher.register_agent(agent2.id, agent2)

    print("Initial Agent Status:", dispatcher.get_agent_status())

    # Simulate dispatching tasks
    tasks_to_dispatch = [
        {"subtask_id": "task_A", "description": "Do thing A"},
        {"subtask_id": "task_B", "description": "Do thing B"},
        {"subtask_id": "task_C", "description": "Do thing C"},
    ]

    assignments = await dispatcher.dispatch_subtasks(tasks_to_dispatch)
    print("Task Assignments:", assignments)
    print("Agent Status after dispatch:", dispatcher.get_agent_status())

    # Simulate waiting for tasks to complete (basic version)
    # In a real system, completion updates would come from agents asynchronously
    await asyncio.sleep(1.5) # Wait longer than task simulation time
    print("Agent Status after simulated completion:", dispatcher.get_agent_status())

    # Stop monitoring
    # await dispatcher.stop_monitoring()


if __name__ == "__main__":
    # To run the example: python -m core.dispatcher
    # Note: Adjust path based on how you run it relative to the project root.
    # Running directly might require adding project root to PYTHONPATH or using relative imports.
    asyncio.run(main())
