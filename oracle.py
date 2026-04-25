class Oracle:
    def __init__(self, horizon=10):
        self.horizon = horizon  # how many steps ahead to predict

    def predict_demand(self, state):
        """
        Predict future request demand for next `horizon` steps.

        Uses simple trend extrapolation:
        growth = current - previous
        """

        curr = state.current_requests
        prev = state.previous_requests

        growth = curr - prev

        predictions = []
        future = curr

        for _ in range(self.horizon):
            future = max(0, future + growth)  # demand can't go negative
            predictions.append(int(future))

        return predictions