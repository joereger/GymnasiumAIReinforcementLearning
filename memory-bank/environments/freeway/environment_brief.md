# Environment Brief: Freeway

## Environment Overview

Freeway is an Atari 2600 game that simulates a chicken crossing a 10-lane highway. The player controls a chicken that must navigate across multiple lanes of traffic, avoiding cars that move at varying speeds. The goal is to reach the other side of the highway to score points.

![Freeway Game Image](https://gymnasium.farama.org/_images/freeway.gif)

## Technical Specifications

- **Environment ID**: `ALE/Freeway-v5`
- **Observation Space**: `Box(0, 255, (210, 160, 3), uint8)` - RGB images of the game screen
- **Action Space**: `Discrete(3)` - 0: NOOP, 1: UP, 2: DOWN
- **Reward Structure**:
  - +1 point for each chicken that successfully crosses all lanes
  - 0 otherwise

## Environment Characteristics

- **Difficulty**: Medium
- **Type**: Deterministic with stochastic elements (car patterns)
- **Key Challenges**:
  1. Sparse rewards (only awarded upon full crossing)
  2. Avoiding multiple moving obstacles (cars)
  3. Planning optimal crossing times/paths
  4. Long time horizon between actions and rewards

## Preprocessing

Standard preprocessing for this environment:
- Convert RGB images to grayscale
- Resize to 84x84 pixels
- Stack 4 consecutive frames to capture motion
- Normalize pixel values to [0,1]

## Reward Engineering Considerations

The default reward structure is sparse (+1 only upon successful crossing), which can make learning difficult. Potential reward engineering approaches:

1. **Positional rewards**: Award partial rewards for upward progress
2. **Zone-based rewards**: Divide the screen into zones and reward zone transitions
3. **Time penalties**: Add small penalties to encourage efficient crossing
4. **Collision penalties**: Add explicit penalties for being hit by cars

## Variants

The environment has multiple variants with different settings:
- `Freeway-v0` through `Freeway-v4`: Older versions with different frame skipping and stochasticity
- `ALE/Freeway-v5`: The current recommended version
- `ALE/Freeway-ram-v5`: RAM-based observations instead of pixel observations

## Reference

- [Gymnasium Documentation for Freeway](https://gymnasium.farama.org/environments/atari/freeway/)
- [Original Atari 2600 Game Manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=198)
