"""Tests for pipeline infrastructure."""

import unittest
from typing import override

from stt_server.pipeline import SingleStage, ChainedStage


class DoubleStage(SingleStage[int, int]):
    """Test stage that doubles input."""

    @override
    async def _process_item(self, item: int) -> None:
        if self._output_queue is not None:
            await self._output_queue.put(item * 2)


class AddOneStage(SingleStage[int, int]):
    """Test stage that adds one to input."""

    @override
    async def _process_item(self, item: int) -> None:
        if self._output_queue is not None:
            await self._output_queue.put(item + 1)


class FilterEvenStage(SingleStage[int, int]):
    """Test stage that only emits even numbers."""

    @override
    async def _process_item(self, item: int) -> None:
        if item % 2 == 0 and self._output_queue is not None:
            await self._output_queue.put(item)


class CollectSink(SingleStage[int, None]):
    def __init__(self, result: list[int]):
        super().__init__()
        self._result: list[int] = result
    
    @override
    async def _process_item(self, item: int) -> None:
        self._result.append(item)


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    """Test cases for pipeline infrastructure."""

    async def test_single_stage(self):
        """Test a single stage processes items correctly."""
        result: list[int] = []
        pipeline = DoubleStage() + CollectSink(result)

        await pipeline.process(5)
        await pipeline.process(10)
        await pipeline.process(None)
        await pipeline.join()

        self.assertEqual(result, [10, 20])

    async def test_two_stage_chain(self):
        """Test chaining two stages with + operator."""
        result: list[int] = []
        pipeline = DoubleStage() + AddOneStage() + CollectSink(result)

        # 5 -> double -> 10 -> add one -> 11
        await pipeline.process(5)
        await pipeline.process(None)
        await pipeline.join()

        self.assertEqual(result, [11])

    async def test_three_stage_chain(self):
        """Test chaining three stages."""
        result: list[int] = []
        pipeline = DoubleStage() + AddOneStage() + DoubleStage() + CollectSink(result)

        await pipeline.process(15)
        await pipeline.process(16)
        await pipeline.process(None)
        await pipeline.join()

        self.assertEqual(result, [62, 66])

    async def test_chain_is_chained_stage(self):
        """Test that + operator returns ChainedStage."""
        stage1 = DoubleStage()
        stage2 = AddOneStage()
        chain = stage1 + stage2

        self.assertIsInstance(chain, ChainedStage)
        await chain.process(None)
        await chain.join()

    async def test_filter_stage(self):
        """Test a stage that filters (doesn't emit for some inputs)."""
        result: list[int] = []
        pipeline = DoubleStage() + FilterEvenStage() + CollectSink(result)

        await pipeline.process(1)  # 1 -> 2 (even, passes)
        await pipeline.process(2)  # 2 -> 4 (even, passes)
        await pipeline.process(3)  # 3 -> 6 (even, passes)
        await pipeline.process(None)
        await pipeline.join()

        self.assertEqual(result, [2, 4, 6])

    async def test_multiple_items(self):
        """Test processing multiple items through a pipeline."""
        result: list[int] = []
        pipeline = DoubleStage() + AddOneStage() + CollectSink(result)

        for i in range(5):
            await pipeline.process(i)
        await pipeline.process(None)
        await pipeline.join()

        # i -> 2*i -> 2*i + 1
        self.assertEqual(result, [1, 3, 5, 7, 9])


if __name__ == "__main__":
    _ = unittest.main()
