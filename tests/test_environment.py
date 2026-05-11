from gesture_dqn.environment import EMGEnvironment


def test_environment_loads_sample_data():
    env = EMGEnvironment(
        "Main/s1/S1_E1_A1.mat",
        window_size=50,
        max_samples=200,
        channels=[0, 1],
    )

    state = env.reset()
    next_state, reward, done = env.step(0, step_size=50)

    assert state.shape == (50, 2)
    assert next_state.shape == (50, 2)
    assert reward in {-1.0, 1.0}
    assert isinstance(done, bool)
