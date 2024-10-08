"""Ravens tasks."""

from cliport.tasks.align_box_corner import AlignBoxCorner
from cliport.tasks.assembling_kits import AssemblingKits
from cliport.tasks.assembling_kits import AssemblingKitsEasy
from cliport.tasks.assembling_kits_seq import AssemblingKitsSeqSeenColors
from cliport.tasks.assembling_kits_seq import AssemblingKitsSeqUnseenColors
from cliport.tasks.assembling_kits_seq import AssemblingKitsSeqFull
from cliport.tasks.block_insertion import BlockInsertion
from cliport.tasks.block_insertion import BlockInsertionEasy
from cliport.tasks.block_insertion import BlockInsertionNoFixture
from cliport.tasks.block_insertion import BlockInsertionSixDof
from cliport.tasks.block_insertion import BlockInsertionTranslation
from cliport.tasks.manipulating_rope import ManipulatingRope
from cliport.tasks.align_rope import AlignRope
from cliport.tasks.packing_boxes import PackingBoxes
from cliport.tasks.packing_shapes import PackingShapes
from cliport.tasks.packing_boxes_pairs import PackingBoxesPairsSeenColors
from cliport.tasks.packing_boxes_pairs import PackingBoxesPairsUnseenColors
from cliport.tasks.packing_boxes_pairs import PackingBoxesPairsFull
from cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsSeq
from cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsGroup
from cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsGroup
from cliport.tasks.palletizing_boxes import PalletizingBoxes
from cliport.tasks.place_red_in_green import PlaceRedInGreen
from cliport.tasks.pick_and_place_primitive import PickAndPlacePrimitive
from cliport.tasks.pick_and_place_primitive import PickAndPlacePrimitiveWithSize
from cliport.tasks.pick_and_place_primitive import PickAndPlacePrimitiveWithRelativePosition
from cliport.tasks.pick_and_place_primitive import PickAndPlacePrimitiveWithAbsolutePosition
# from cliport.tasks.put_block_in_bowl import PutBlockInBowlSeenColors
# from cliport.tasks.put_block_in_bowl import PutBlockInBowlUnseenColors
# from cliport.tasks.put_block_in_bowl import PutBlockInBowlFull
from cliport.tasks.put_block_in_bowl import (
    PutBlockInMatchingBowl, PutBlockInMismatchingBowl,
    PutBlockIntoMatchingBowlWithDetails
)
from cliport.tasks.put_block_in_bowl import (
    PutHiddenBlockIntoMatchingBowl,
    PutHiddenBlocksInTwoLayerTowersIntoMatchingBowls,
    PutHiddenBlocksInThreeLayerTowersIntoMatchingBowls,
    PutHiddenBlocksInPyramidIntoMatchingBowls,
)
from cliport.tasks.put_block_in_bowl import PutAllBlockInABowl
from cliport.tasks.put_block_in_bowl import PutAllBlockOnCorner
# from cliport.tasks.put_block_in_bowl import PutAllBlockInAZone
from cliport.tasks.put_block_in_bowl import PickAndPlace
from cliport.tasks.stack_block_pyramid import StackBlockPyramid
from cliport.tasks.stack_blocks import StackBlockPyramidSeqSeenColors
from cliport.tasks.stack_blocks import StackBlockPyramidSeqUnseenColors
from cliport.tasks.stack_blocks import StackBlockPyramidSeqFull
from cliport.tasks.stack_blocks import StackBlockPyramidWithoutSeq
from cliport.tasks.stack_blocks import (StackAllBlocksOnAZone, StackAllBlocksOnAZoneWithDetails)
from cliport.tasks.stack_blocks import StackBlocksWithAlternateColor
from cliport.tasks.stack_blocks import (
    StackBlockOfSameColor, StackBlocksOfSameSize,
    StackBlocksByColor, StackBlocksByColorAndSize, StackBlocksByColorInSizeOrder,
)
# from cliport.tasks.stack_block_pyramid_seq import StackBlocksOfCoolColors
# from cliport.tasks.stack_block_pyramid_seq import StackBlocksOfWarmColors
# from cliport.tasks.stack_block_pyramid_seq import StackBlocksOfPrimaryColors
# from cliport.tasks.stack_block_pyramid_seq import StackBlocksOfSecondaryColors
# from cliport.tasks.stack_block_pyramid_seq import (
#     StackBiggerCoolBlocks, StackSmallerCoolBlocks, StackBiggerWarmBlocks,
#     StackSmallerWarmBlocks, StackBiggerPrimaryBlocks, StackSmallerPrimaryBlocks,
#     StackBiggerSecondaryBlocks, StackSmallerSecondaryBlocks
# )
# from cliport.tasks.stack_block_pyramid_seq import (
#     StackBlocksOfCoolColorInSizeOrder, StackBlocksOfWarmColorInSizeOrder,
#     StackBlocksOfPrimaryColorInSizeOrder, StackBlocksOfSecondaryColorInSizeOrder
# )
from cliport.tasks.stack_blocks import (
    StackBlocksByRelativePositionAndColor,
    StackBlocksByRelativePositionAndColorAndSize,
    StackBlocksByAbsolutePositionAndColorInSizeOrder,
    StackBlocksByAbsolutePositionAndColorAndSize,
)

from cliport.tasks.move_blocks import (
    MoveBlocksBetweenAbsolutePositions,
    MoveBlocksBetweenAbsolutePositionsByColor,
    MoveBlocksBetweenAbsolutePositionsBySize,
    MoveBlocksBetweenAbsolutePositionsBySizeAndColor,
)

from cliport.tasks.sweeping_piles import SweepingPiles
from cliport.tasks.separating_piles import SeparatingPilesSeenColors
from cliport.tasks.separating_piles import SeparatingPilesUnseenColors
from cliport.tasks.separating_piles import SeparatingPilesFull
from cliport.tasks.task import Task
from cliport.tasks.towers_of_hanoi import TowersOfHanoi
from cliport.tasks.towers_of_hanoi_seq import TowersOfHanoiSeqSeenColors
from cliport.tasks.towers_of_hanoi_seq import TowersOfHanoiSeqUnseenColors
from cliport.tasks.towers_of_hanoi_seq import TowersOfHanoiSeqFull
from cliport.tasks.size_reasoning_tasks import StackSmallerOverBiggerWithSameColor
from cliport.tasks.size_reasoning_tasks import StackSmallerOverBiggerWithSameColorInSameColorZone
from cliport.tasks.arithmetic_reasoning import PutEvenBlockInSameColorZone

names = {
    # demo conditioned
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi,

    # goal conditioned
    'align-rope': AlignRope,
    'assembling-kits-seq-seen-colors': AssemblingKitsSeqSeenColors,
    'assembling-kits-seq-unseen-colors': AssemblingKitsSeqUnseenColors,
    'assembling-kits-seq-full': AssemblingKitsSeqFull,
    'packing-shapes': PackingShapes,
    'packing-boxes-pairs-seen-colors': PackingBoxesPairsSeenColors,
    'packing-boxes-pairs-unseen-colors': PackingBoxesPairsUnseenColors,
    'packing-boxes-pairs-full': PackingBoxesPairsFull,
    'packing-seen-google-objects-seq': PackingSeenGoogleObjectsSeq,
    'packing-unseen-google-objects-seq': PackingUnseenGoogleObjectsSeq,
    'packing-seen-google-objects-group': PackingSeenGoogleObjectsGroup,
    'packing-unseen-google-objects-group': PackingUnseenGoogleObjectsGroup,
    # 'put-block-in-bowl-seen-colors': PutBlockInBowlSeenColors,
    # 'put-block-in-bowl-unseen-colors': PutBlockInBowlUnseenColors,
    # 'put-block-into-bowl-full': PutBlockInBowlFull,
    'put-block-into-matching-bowl': PutBlockInMatchingBowl,
    'put-block-into-mismatching-bowl': PutBlockInMismatchingBowl,
    'put-blocks-into-matching-bowls-with-details': PutBlockIntoMatchingBowlWithDetails,
    'put-hidden-block-into-matching-bowl': PutHiddenBlockIntoMatchingBowl,
    'put-hidden-blocks-in-two-layer-towers-into-matching-bowls':
        PutHiddenBlocksInTwoLayerTowersIntoMatchingBowls,
    'put-hidden-blocks-in-three-layer-towers-into-matching-bowls':
        PutHiddenBlocksInThreeLayerTowersIntoMatchingBowls,
    # 'put-hidden-blocks-in-matching-bowls': PutHiddenBlocksInMatchingBowls,
    'put-hidden-blocks-in-pyramid-into-matching-bowls': PutHiddenBlocksInPyramidIntoMatchingBowls,
    'put-all-block-in-a-bowl': PutAllBlockInABowl,
    'put-all-block-on-corner': PutAllBlockOnCorner,
    # 'put-all-block-in-a-zone': PutAllBlockInAZone,
    'pick-and-place': PickAndPlace,
    'pick-and-place-primitive': PickAndPlacePrimitive,
    'pick-and-place-primitive-with-size': PickAndPlacePrimitiveWithSize,
    'stack-block-pyramid-seq-seen-colors': StackBlockPyramidSeqSeenColors,
    'stack-block-pyramid-seq-unseen-colors': StackBlockPyramidSeqUnseenColors,
    'stack-block-pyramid-seq-full': StackBlockPyramidSeqFull,
    'stack-block-pyramid-without-seq': StackBlockPyramidWithoutSeq,
    'stack-all-blocks-on-a-zone': StackAllBlocksOnAZone,
    'stack-all-blocks-on-a-zone-with-details': StackAllBlocksOnAZoneWithDetails,
    # 'stack-all-block-of-same-color': StackAllBlockOfSameColor,
    'stack-blocks-with-alternate-color': StackBlocksWithAlternateColor,
    'separating-piles-seen-colors': SeparatingPilesSeenColors,
    'separating-piles-unseen-colors': SeparatingPilesUnseenColors,
    'separating-piles-full': SeparatingPilesFull,
    'towers-of-hanoi-seq-seen-colors': TowersOfHanoiSeqSeenColors,
    'towers-of-hanoi-seq-unseen-colors': TowersOfHanoiSeqUnseenColors,
    'towers-of-hanoi-seq-full': TowersOfHanoiSeqFull,
    'stack-blocks-of-same-color': StackBlockOfSameColor,
    'stack-blocks-by-color': StackBlocksByColor,
    'stack-blocks-by-color-and-size': StackBlocksByColorAndSize,
    'stack-blocks-by-color-in-size-order': StackBlocksByColorInSizeOrder,
    'stack-blocks-of-same-size': StackBlocksOfSameSize,
    'stack-smaller-over-bigger-with-same-color': StackSmallerOverBiggerWithSameColor,
    'stack-smaller-over-bigger-with-same-color-in-same-color-zone':
        StackSmallerOverBiggerWithSameColorInSameColorZone,
    'stack-blocks-by-relative-position-and-color': StackBlocksByRelativePositionAndColor,
    'stack-blocks-by-relative-position-and-color-and-size':
        StackBlocksByRelativePositionAndColorAndSize,
    'stack-blocks-by-absolute-position-and-color-in-size-order':
        StackBlocksByAbsolutePositionAndColorInSizeOrder,
    'stack-blocks-by-absolute-position-and-color-and-size':
        StackBlocksByAbsolutePositionAndColorAndSize,
    'pick-and-place-primitive-with-relative-position': PickAndPlacePrimitiveWithRelativePosition,
    'pick-and-place-primitive-with-absolute-position': PickAndPlacePrimitiveWithAbsolutePosition,
    'put-even-blocks-in-same-color-zone': PutEvenBlockInSameColorZone,
    'move-blocks-between-absolute-positions': MoveBlocksBetweenAbsolutePositions,
    'move-blocks-between-absolute-positions-by-color': MoveBlocksBetweenAbsolutePositionsByColor,
    'move-blocks-between-absolute-positions-by-size': MoveBlocksBetweenAbsolutePositionsBySize,
    'move-blocks-between-absolute-positions-by-size-and-color':
        MoveBlocksBetweenAbsolutePositionsBySizeAndColor,
}
