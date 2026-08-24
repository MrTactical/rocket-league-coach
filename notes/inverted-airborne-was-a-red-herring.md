# The "62% inverted airborne" figure was measuring the wrong thing

Dated 2026-08-23. Written after the collision meshes were dumped, which is what
finally made this checkable.

## The claim I had been chasing

Live telemetry said the cars spent ~62% of their airborne time with the roof
pointing away from upright. I treated that as a recovery-controller defect and
spent three separate searches on it.

## Why it never resolved

Every search came back clean, which should have been the clue:

1. `tools/optimise.py` on orientation gains: already near optimal (0.66s to
   right, 12/12 attitudes, only +3.2% available).
2. `tools/optimise.py` on dodge landing: 0/15 inverted in free space.
3. `tools/wallcheck.py` (new, mesh-backed): the SHIPPED `Recovery` controller
   lands wheels-down in every zone, walls and corners included.

   | zone           | settled | 1st touchdown align | bad landings |
   |----------------|---------|---------------------|--------------|
   | open floor     | 96%     | 0.98                | 1%           |
   | near side wall | 100%    | 0.94                | 0%           |
   | at side wall   | 100%    | 0.98                | 0%           |
   | corner         | 100%    | 0.97                | 0%           |
   | back wall      | 100%    | 0.99                | 0%           |

   Walls were my main hypothesis, since `predict_landing` says "ignoring walls"
   in its own docstring and `Recovery` always levels to world up. Both are
   indeed wrong near geometry, and it turns out not to matter: the wall zones
   settle FASTER than open floor, because a surface arrives sooner.

## What is actually happening

Bucketing every airborne episode in the traces by duration:

| duration      | episodes | airborne ticks | share of airborne time |
|---------------|----------|----------------|------------------------|
| 0.00 - 0.10 s | 451      | 2757           | 20%                    |
| 0.10 - 0.25 s | 489      | 8496           | 61%                    |
| 0.25 - 0.50 s | 65       | 2365           | 17%                    |
| 0.50 - 1.00 s | 2        | 125            | 1%                     |
| over 1.00 s   | 1        | 175            | 1%                     |

Righting an inverted car takes ~0.60s. **99% of airborne time is in episodes
shorter than that**, and those episodes hold 98% of the inverted ticks. Exactly
one episode in 1008 was long enough to recover in, and it ended upright.

So the cars were never failing to recover. They were being knocked askew by
ball contact and landing bounces, spending 0.1-0.2s in the air, and touching
down again before any controller could possibly correct. The metric was
counting physics, not a fault.

Two confounds also had to be removed to see this, both of which inflated the
original number:
  * 84% of the not-upright GROUNDED samples are cars driving on walls, where
    `up.z` is near 0 by definition and nothing is wrong.
  * `wheels_with_contact` is a 4-tuple of bools, not a count, and
    `has_world_contact` means chassis contact, not wheel contact. Getting
    either wrong makes a working recovery look broken. `tools/simrs.py`
    provides `nearest_surface()` to judge alignment against whatever surface
    the car is actually on.

## The real finding

Only 3 of 1008 airborne episodes exceeded 0.5s. The bots essentially never get
airborne for a meaningful length of time. That is the same underlying fact as
"aerials are 1-2%" and the user's "they never fly" -- not a separate problem.

## What not to do next

Do not spend more effort on recovery or orientation tuning. Three independent
measurements say that layer is sound. The cost is in the decision to commit to
leaving the ground at all.
