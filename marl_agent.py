import numpy as np


class PolicyNetwork:
    def __init__(self, input_dim, hidden_dim=16, lr=0.001):
        self.lr = lr

        # weights
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(3, hidden_dim) * 0.1
        self.b2 = np.zeros(3)

    def forward(self, x):
        self.h = np.tanh(self.W1 @ x + self.b1)
        logits = self.W2 @ self.h + self.b2
        return logits

    def softmax(self, logits):
        exp = np.exp(logits - np.max(logits))
        return exp / np.sum(exp)

    def sample_action(self, x):
        logits = self.forward(x)
        probs = self.softmax(logits)
        probs[1] *= 0.9      # slightly reduce "do nothing"
        probs[0] *= 1.05     # slightly increase scale down
        probs[2] *= 1.05     # slightly increase scale up
        probs = probs / np.sum(probs)
        action_idx = np.random.choice(3, p=probs)
        action = action_idx - 1  # map {0,1,2} → {-1,0,1}

        return action, probs, action_idx

    def update(self, x, action_idx, advantage):
        logits = self.forward(x)
        probs = self.softmax(logits)

        # output gradient
        dlogits = -probs
        dlogits[action_idx] += 1
        dlogits *= advantage

        entropy = -np.sum(probs * np.log(probs + 1e-8))
        dlogits *= (advantage + 0.01 * entropy)

        # gradients for second layer
        dW2 = np.outer(dlogits, self.h)
        db2 = dlogits

        # backprop into hidden
        dh = self.W2.T @ dlogits
        dh *= (1 - self.h ** 2)  # tanh derivative

        # gradients for first layer
        dW1 = np.outer(dh, x)
        db1 = dh

        # update
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2

        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1