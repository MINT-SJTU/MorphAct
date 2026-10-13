"""Episode-level seeding for the MorphAct evaluation server."""

from morphact.eval_seed import EpisodeSeed, SEED_SCHEME


class EpisodeSeededPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.episode_seed = EpisodeSeed()

    @property
    def metadata(self):
        return {**self.policy.metadata, "episode_seed_protocol": SEED_SCHEME}

    def infer(self, observation):
        observation = dict(observation)
        self.episode_seed.begin(
            observation.pop("_eval_episode_id"), observation.pop("_eval_policy_seed")
        )
        return self.policy.infer(observation)
