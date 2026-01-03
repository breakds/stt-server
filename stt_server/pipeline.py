"""Pipeline infrastructure for streaming audio processing.

A pipeline consists of stages that process items asynchronously. Each stage
runs its own task, pulling from an input queue and emitting to an output queue.

Stages can be composed using the + operator:
    pipeline = stage1 + stage2 + stage3

Usage:
    # Define a stage
    class MyStage(SingleStage[InputType, OutputType]):
        async def _process_item(self, item: InputType) -> None:
            result = transform(item)
            if self._output_queue is not None:
                await self._output_queue.put(result)

    # Compose stages
    pipeline = Stage1() + Stage2() + Stage3()

    # Wire output (for ProcessChain, set on the last stage)
    output_queue = asyncio.Queue()
    pipeline._last._output_queue = output_queue

    # Push items
    await pipeline.process(item)

    # Shutdown
    await pipeline.process(None)  # Sentinel propagates through
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, override, cast

In = TypeVar("In")
Out = TypeVar("Out")
T = TypeVar("T")


class Stage(ABC, Generic[In, Out]):
    """Abstract base class for pipeline stages.

    Both SingleStage and ProcessChain inherit from this, providing a uniform
    interface for composition and processing.
    """

    def __init__(self,
                 input_queue: asyncio.Queue[In | None]):
        self._input_queue: asyncio.Queue[In | None] = input_queue
        self._output_queue: asyncio.Queue[Out | None] | None = None

    @abstractmethod
    async def process(self, item: In | None) -> None:
        """Push an item into this stage for processing.

        Args:
            item: The item to process, or None as a shutdown sentinel.
        """
        pass

    @abstractmethod
    def __add__(self, other: Stage[Out, T]) -> ChainedStage[In, T]:
        """Chain this stage with another: stage1 + stage2

        Args:
            other: The next stage in the chain.

        Returns:
            A ProcessChain representing the composition.
        """
        pass

    @abstractmethod
    async def join(self):
        pass


class SingleStage(Stage[In, Out], ABC):
    """A single processing stage with its own queue and async task.

    Subclasses implement _process_item() to define the processing logic.
    The stage automatically runs a task that pulls from the input queue
    and calls _process_item() for each item.
    """
    def __init__(self):
        super().__init__(asyncio.Queue())
        self._task: asyncio.Task[None] = asyncio.create_task(self._run())

    @override
    async def process(self, item: In | None) -> None:
        await self._input_queue.put(item)

    @abstractmethod
    async def _process_item(self, item: In) -> None:
        """Process one input item.

        Subclasses implement this method. Call _emit() zero or more times
        to produce output items.

        Args:
            item: The input item to process.
        """
        pass

    @override
    async def join(self):
        await self._task


    @override
    def __add__(self, other: Stage[Out, T]) -> ChainedStage[In, T]:
        """Chain this stage with another: stage1 + stage2

        Args:
            other: The next stage in the chain.

        Returns:
            A ProcessChain representing the composition.
        """
        self._output_queue: asyncio.Queue[Out | None] | None = other._input_queue
        return ChainedStage(cast(Stage[In, object], self), cast(Stage[object, T], other))

    async def _run(self) -> None:
        """Main loop: pull from input, process, repeat until sentinel."""
        while (item := await self._input_queue.get()) is not None:
            await self._process_item(item)
        if self._output_queue is not None:
            await self._output_queue.put(None)


class ChainedStage(Stage[In, Out]):
    """A chain of stages that acts as a single stage.

    Created by composing stages with the + operator. Delegates process()
    to the first stage and set_output() to the last stage.
    """

    def __init__(self, first: Stage[In, object], last: Stage[object, Out]):
        super().__init__(first._input_queue)
        self._first: Stage[In, object] = first
        self._last: Stage[object, Out] = last

    @override
    async def process(self, item: In | None) -> None:
        await self._first.process(item)

    @override
    async def join(self):
        await self._first.join()
        await self._last.join()

    @override
    def __add__(self, other: Stage[Out, T]) -> ChainedStage[In, T]:
        """Chain this stage with another: stage1 + stage2

        Args:
            other: The next stage in the chain.

        Returns:
            A ProcessChain representing the composition.
        """
        self._last._output_queue = other._input_queue
        return ChainedStage(cast(Stage[In, object], self), cast(Stage[object, T], other))
