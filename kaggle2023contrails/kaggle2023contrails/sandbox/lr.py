import torch
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import matplotlib.pyplot as plt

model = torch.nn.Linear(10, 5)
optimizer = torch.optim.SGD(model.parameters(), lr=100)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps=264,
    num_training_steps=6600,
    num_cycles=1,
)

lrs = []
for i in range(6600):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(lrs)
plt.show()

