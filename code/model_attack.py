import torch
import numpy as np
from aggregation import multi_krum

def multi_krum(all_updates, n_attackers, multi_k=False):

    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    aggregate = torch.mean(candidates, dim=0)

    return aggregate, np.array(candidate_indices) 





def attacking_trimmed_mean(all_updates, deviation, n_attackers):
    b = 2
    max_vector = torch.max(all_updates, 0)[0]
    min_vector = torch.min(all_updates, 0)[0]

    max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
    min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
    min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    mal_updates = torch.cat((mal_vec, all_updates), 0)
    # print(mal_updates.shape)

    return mal_updates


def compute_lambda(all_updates, model_re, n_attackers):
    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1)
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0])
    max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])

    return (term_1 + max_wre_dist)


def attacking_krum(all_updates, model_re, deviation, n_attackers):
    lamda = compute_lambda(all_updates, model_re, n_attackers)
    threshold = 1e-5

    mal_updates = []
    while lamda > threshold:
        mal_update = (- lamda * deviation)

        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=False)

        if krum_candidate < n_attackers:
            return mal_updates

        lamda *= 0.5

    if not len(mal_updates):
        print(lamda, threshold)
        mal_update = (model_re - lamda * deviation)

        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates