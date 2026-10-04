# Mocap CSV to pose `.npy`

Checked on `mocap_raw_example.csv` and `mocap_pose_example.npy`. Frame 0, 100, 5000, and the last frame match exactly.

The CSV is an OptiTrack export: 120 Hz, global coordinates, millimeters, quaternions. It already contains solved bone positions. The conversion copies 20 of those positions. It does not run forward kinematics, and it does not use quaternions, finger bones, the 50 raw markers, or the `Node1`–`Node5` rigid bodies.

## Positions

Take each bone's `Position` columns (`X, Y, Z`), in this order. `Ab` is skipped.

1. Skeleton root (`Skeleton-Aaron`)
2. Chest, Neck, Head
3. LShoulder, LUArm, LFArm, LHand
4. RShoulder, RUArm, RFArm, RHand
5. LThigh, LShin, LFoot, LToe
6. RThigh, RShin, RFoot, RToe

Stack every exported frame into `positions` with shape `(F, 20, 3)`, `float64`, still in millimeters. This example has `F = 11477`, the same as `Total Exported Frames`.

## Timestamps

`timestamps[i]` is the header `Capture Start Time` plus the row's `Time (Seconds)`, written as an ISO-8601 string (`2024-05-18T13:43:59.273000`). At 120 Hz the step is 8 ms.

## File

`np.save` a dict:

- `timestamps`: shape `(F,)`
- `positions`: shape `(F, 20, 3)`
