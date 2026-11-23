# Fixed split (decided before any training)

Image names: card_001.jpg to card_070.jpg

- Test (7): card_010, card_020, card_030, card_040, card_050, card_060, card_070
- Val (14): card_003, card_007, card_013, card_017, card_023, card_027, card_033, card_037, card_043, card_047, card_053, card_057, card_063, card_067
- Train (49): all remaining images

Seed: 42
Test images are never used for training, tuning or picking a checkpoint.