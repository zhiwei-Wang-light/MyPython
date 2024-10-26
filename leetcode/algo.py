import heapq
from collections import Counter


def topKFrequent(nums, k):
    counter = Counter(nums)
    heaq = []
    for key, value in counter.items():
        if len(heaq) < k:
            heapq.heappush(heaq, (value, key))
        else:
            if heaq[0][0] > value:
                pass
            else:
                heapq.heappop(heaq)
                heapq.heappush(heaq, (value, key))
    return [key for value, key in heaq]
