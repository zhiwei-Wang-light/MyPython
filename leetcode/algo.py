import collections
import heapq
from collections import deque, defaultdict, Counter
import math


def sum_of_two_numbers(nums, target):
    """
    前缀和+哈希表
    """
    hashmap = {}
    cnt = 0
    for i, n in enumerate(nums):
        # 在历史出现的数字中存在m+n=target
        if target - n in hashmap.keys():
            cnt += 1
        # 使用字典记录数字对应的索引
        hashmap[n] = i
    return cnt


def max_in_sliding_window(nums, k):
    """
    找出数组中大小为k的滑动窗口中的最大值。使用队列,维持固定长度

    :param nums: List[int] 输入的整数数组
    :param k: int 滑动窗口的大小
    :return: List[int] 滑动窗口中的最大值序列
    """
    # 如果是空列表返回空
    if not len(nums):
        return []
    max_value = []
    q = deque()
    for i in range(len(nums)):
        while q and q[0] < i - k + 1:
            q.popleft()
        while q and nums[q[-1]] < nums[i]:
            q.pop()
        q.append(i)
        if i >= k - 1:
            max_value.append(nums[q[0]])

    return max_value


def maximum_subarray_sum(nums):
    ans = 0
    max_value = 0
    for n in nums:
        ans += n
        ans = max(ans, 0)
        max_value = max(max_value, ans)
    return max_value


def merge_intervals(lists):
    lists = sorted(lists, key=lambda x: x[0])
    merged = []
    for l in lists:
        if len(merged) and merged[-1][-1] >= l[0]:
            merged[-1][-1] = max(merged[-1][-1], l[-1])
        else:
            merged.append(l)
    return merged


def nums_of_k(nums, k):
    """
    前缀和+哈希表,使用defaultdict,如果key不存在返回0
    :param nums:
    :param k:
    :return:
    """
    default_map = defaultdict(int)
    # 手动添加
    default_map[0] = 1
    ans = 0
    cnt = 0
    for n in nums:
        ans += n
        cnt += default_map[ans - k]
        default_map[ans] += 1
    return cnt


def mainWindow(s, t):
    ans_left = -1
    ans_right = len(s)
    cnt_s = Counter()
    cnt_t = Counter(t)
    # 共有less种不同的字母还需要在s中找到
    less = len(cnt_t)
    left = 0
    for right, c in enumerate(s):
        cnt_s[c] += 1
        if cnt_s[c] == cnt_t[c]:
            less -= 1
        # 找到含有s中所有字母的子串,但不是最小的子串
        while less == 0:
            # 寻找子串长度
            if right - left < ans_right - ans_left:
                ans_left = left
                ans_right = right
            if cnt_s[s[left]] == cnt_t[s[left]]:
                less += 1
            cnt_s[s[left]] -= 1
            left += 1
    return "" if ans_right < 0 else s[ans_left:ans_right + 1]


def max_sub_string_without_cover(s):
    """
    最大不重叠子串,更新左边界,更新最大长度
    Args:
        s:字符串

    Returns:
        最大不重叠子串长度

    Examples:
        >>> c=max_sub_string_without_cover("asajfda")

    """
    q = collections.deque()
    left = 0
    max_length = 0
    for right, c in enumerate(s):
        while c in q:
            q.popleft()
            left += 1
        max_length = max(max_length, right - left + 1)
        q.append(c)
    return max_length


def dot_without_self(nums):
    n = len(nums)
    pre = [0] * n
    pre[0] = 1
    ans = 1
    for i in range(1, n):
        pre[i] = pre[i - 1] * nums[i - 1]
    for j in reversed(range(n - 1)):
        ans = ans * nums[j + 1]
        pre[j] = pre[j] * ans
    return pre


def group_anagrams(words):
    """
    异位词分组结果
    """
    # 初始化字典，字典如果没有该key，则添加key，value为[]
    hash_map = defaultdict(list)
    for word in words:
        # 异位词排序结果相同，将异位词排序后作为key
        hash_map["".join(c for c in sorted(word))].append(word)
    return hash_map.values()


def longest_consecutive_sequence(nums):
    """
    使用一个变量累计长度

    """
    nums = set(nums)
    max_length = 0
    for n in nums:
        if n - 1 not in nums:
            cnt = 0
            while n in nums:
                cnt += 1
                n += 1
            max_length = max(max_length, cnt)
    return max_length


def swap(left, right, nums):
    tmp = nums[left]
    nums[left] = nums[right]
    nums[right] = tmp


def move0(nums):
    left = 0
    for right in range(len(nums)):
        if nums[right]:
            swap(left, right, nums)
            left += 1
    return nums


def max_water(nums):
    left = 0
    right = len(nums) - 1
    max_water = 0
    while left < right:
        max_water = max(min(nums[right], nums[left]) * (right - left - 1), max_water)
        if nums[left] < nums[right]:
            left += 1
        else:
            right -= 1
    return max_water


def collecting_rainwater(nums):
    waters = 0
    max_left = []
    max_right = []
    max_left.append(nums[0])
    max_right.append(nums[-1])
    for n in range(1, len(nums)):
        max_left.append(max(nums[n], max_left[-1]))
    for n in reversed(range(len(nums) - 1)):
        max_right.append(max(nums[n], max_left[-1]))
    max_right = max_right[::-1]
    for i in range(len(nums)):
        waters += (min(max_left[i] - nums[i], max_right[i]))
    return waters


def three_sum(nums, target):
    """
    注意列表访问次数过多
    Args:
        nums:
        target:

    Returns:

    """
    nums = sorted(nums)
    n = len(nums)
    ans = []
    for frist in range(n):
        if frist > 0 and nums[frist] == nums[frist - 1]:
            continue
        a = nums[frist]
        for second in range(frist + 1, n):
            if second > frist + 1 and nums[second] == nums[second - 1]:
                continue
            third = n - 1
            b = nums[second]
            while second < third:
                while second < third and a + b + nums[third] - target > 0:
                    third -= 1
                if second < third and a + b + nums[third] - target == 0:
                    ans.append([a, b, nums[third]])
                    break
                else:
                    break
    return ans


def reverse(nums):
    left = 0
    right = len(nums) - 1
    while left < right:
        tmp = nums[left]
        nums[left] = nums[right]
        nums[right] = tmp
        left += 1
        right -= 1
    return nums


def lunzhuanshuzu(nums, k):
    reverse(nums)
    nums[:k] = reverse(nums[:k])
    nums[k:] = reverse(nums[k:])
    return nums


def miss_frist(nums):
    """
    注意小于等于0的全赋值为n+1
    Args:
        nums:

    Returns:

    """
    n = len(nums)
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n + 1
    for i in range(n):
        num = abs(nums[i])
        if num <= n:
            if nums[num - 1] > 0:
                nums[num - 1] *= -1
    for i in range(n):
        if nums[i] > 0:
            return i + 1


def finding_anagrams(s, t):
    """
    固定滑块长度,然后每次删除左边的新增右边的,保持长度不变
    Args:
        s:
        t:

    Returns:

    """
    cnt_s = Counter()
    cnt_t = Counter(t)
    res = []
    for i in range(len(t)):
        cnt_s[s[i]] += 1
    if cnt_s == cnt_t:
        res.append(0)
    for i in range(len(t), len(s)):
        cnt_s[s[i - len(t)]] -= 1
        cnt_s[s[i]] += 1
        if cnt_s[s[i - len(t)]] == 0:
            del cnt_s[s[i - len(t)]]
        if cnt_s == cnt_t:
            res.append(i - len(t) + 1)
    return res


def nums_of_islands(grid):
    def in_area(grid, r, c):
        return 0 <= r < len(grid) and 0 <= c < len(grid[0])

    def dfs(grid, r, c):
        if not in_area(grid, r, c):
            return
        if grid[r][c] != 1:
            return
        grid[r][c] = 0
        dfs(grid, r - 1, c)
        dfs(grid, r + 1, c)
        dfs(grid, r, c - 1)
        dfs(grid, r, c + 1)

    cnt = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                cnt += 1
                dfs(grid, i, j)
    return cnt


def add(i):
    if i == 0:
        return 0
    ret = add(i - 1)
    return ret + i


def swap2(nums, i, j):
    tmp = nums[i]
    nums[i] = nums[j]
    nums[j] = tmp


def permuate(nums, k, m):
    if k == m:
        print(nums)
    for i in range(k, m):
        swap2(nums, k, i)
        permuate(nums, k + 1, m)
        swap2(nums, k, i)


def matrix_zero(lists):
    frist_row_flag = any(l == 0 for l in lists[0])
    frist_col_flag = any(l[0] == 0 for l in lists)
    for i in range(1, len(lists)):
        for j in range(1, len(lists[0])):
            if lists[i][j] == 0:
                lists[i][0] = 0
                lists[0][j] = 0
    for i in range(1, len(lists)):
        for j in range(1, len(lists[0])):
            if lists[i][0] == 0 or lists[0][j] == 0:
                lists[i][j] = 0
    if frist_row_flag:
        for j in range(len(lists[0])):
            lists[0][j] = 0
    if frist_col_flag:
        for i in range(len(lists)):
            lists[i][0] = 0
    return lists


def rotate_matrix(matrix):
    n = len(matrix)
    for row in range(len(matrix) // 2):
        for col in range(len(matrix[0]) // 2):
            matrix[row][col], matrix[n - col - 1][row], matrix[n - row - 1][n - col - 1], matrix[col][n - row - 1] = \
                matrix[n - col - 1][row], matrix[n - row - 1][n - col - 1], matrix[col][n - row - 1], matrix[row][col]
    return matrix


def spiral_matrix(matrix):
    ans = []
    l = 0
    t = 0
    b = len(matrix) - 1
    r = len(matrix[0]) - 1
    while l < r:
        for i in range(l, r + 1):
            ans.append(matrix[t][i])
        t += 1
        for i in range(t, b + 1):
            ans.append(matrix[i][r])
        r -= 1
        for i in reversed(range(l, r + 1)):
            ans.append(matrix[b][i])
        b -= 1
        for i in reversed(range(t, b + 1)):
            ans.append(matrix[i][l])
        l += 1
    return ans


def subset(nums):
    arr = []
    path = []

    def backtrack(nums, index, n):
        path.append(arr[:])
        for i in range(index, n):
            arr.append(nums[i])
            backtrack(nums, i + 1, n)
            arr.pop()

    backtrack(nums, 0, len(nums))
    return path


letters_set = []
word = []


def letter_combination(index, logits):
    letters_map = {2: "abc", 3: "def", 4: "ghi", 5: "jkl", 6: "mno", 7: "pqrs", 8: "tuv", 9: "wxyz"}
    if index == len(logits):
        letters_set.append("".join(x for x in word))
        return
    map = letters_map[logits[index]]
    for m in map:
        word.append(m)
        letter_combination(index + 1, logits)
        word.pop()


def phone_number_letter_combination(index, logits):
    letter_combination(index, logits)
    return letters_set


class Island():
    def __init__(self):
        self.ans = 0

    def dfs(self, lists, row, col):
        if row < 0 or row >= len(lists) or col < 0 or col >= len(lists[0]):
            return 1
        if lists[row][col] == 0:
            return 1
        if lists[row][col] == -1:
            return 0
        lists[row][col] = -1

        a = self.dfs(lists, row - 1, col)
        b = self.dfs(lists, row + 1, col)
        c = self.dfs(lists, row, col - 1)
        d = self.dfs(lists, row, col + 1)
        return a + b + c + d


def island_perimeter(lists):
    island = Island()
    ans = island.dfs(lists, 0, 2)
    return ans


class CombinationSum():
    def __init__(self, target, candidates):
        self.path = []
        self.arr = []
        self.ans = 0
        self.target = target
        self.candidates = candidates

    def dfs(self, candidates, index):
        if index == len(candidates) or self.ans > self.target:
            return
        if self.ans == self.target:
            self.path.append(self.arr[:])
        for i in range(index, len(candidates)):
            self.ans += candidates[i]
            self.arr.append(candidates[i])
            self.dfs(candidates, i)
            self.arr.pop()
            self.ans -= candidates[i]

    def get(self):
        self.dfs(self.candidates, 0)
        return self.path


class GenerateParentheses():
    def __init__(self):
        self.arr = []
        self.path = []

    def generate_parentheses(self, left, right, n):
        if left + right == n * 2:
            self.path.append("".join(s for s in self.arr[:]))
            return
        if left < n:
            self.arr.append('(')
            self.generate_parentheses(left + 1, right, n)
            self.arr.pop()
        if right < left:
            self.arr.append(')')
            self.generate_parentheses(left, right + 1, n)
            self.arr.pop()

    def gp(self, left, right, n):
        self.generate_parentheses(left, right, n)
        return self.path


class WordSearch():
    def __init__(self):
        self.falg = False
        self.arr = []
        self.word = ""

    def word_search(self, grid, row, col, n, index):
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != self.word[index]:
            return False
        if index == len(self.word) - 1:
            return True
        tmp = grid[row][col]
        grid[row][col] = ""
        res = self.word_search(grid, row - 1, col, n, index + 1) or self.word_search(grid, row + 1, col, n,
                                                                                     index + 1) or self.word_search(
            grid, row, col - 1, n, index + 1) or self.word_search(grid, row, col + 1, n, index + 1)
        grid[row][col] = tmp
        return res

    def ws(self, grid, word):
        self.word = word
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if self.word_search(grid, i, j, 0, 0):
                    return True
        return False


class PalindromePartitioning():
    """
    找到一个字符串中的所有回文子串
    例.
    "aab" -> [['a', 'a', 'b'], ['aa', 'b']]
    """
    def __init__(self):
        self.flag = None
        self.arr = []
        self.path = []

    def pp(self, word):
        n = len(word)
        self.flag = [[True] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                self.flag[i][j] = (word[i] == word[j] and self.flag[i + 1][j - 1])
        self.backtrack(word, 0, n)
        return self.path

    def backtrack(self, word, index, n):
        if index == n:
            self.path.append(self.arr[:])
            return
        for i in range(index, n):
            if self.flag[index][i]:
                self.arr.append(word[index:i + 1])
                self.backtrack(word, i + 1, n)
                self.arr.pop()


class NQueen():
    """
    “N皇后”问题是一个经典的回溯算法问题，它要求在一个 N×N 的棋盘上放置 N 个皇后，使得任何一个皇后都无法直接吃掉其他的皇后，即任意两个皇后都不能处于同一行、同一列或同一斜线上。
    给定一个整数 n，返回所有不同的 N 皇后问题的解决方案。每一种解法包含一个明确的 N 皇后问题的棋子放置方案，该方案中 ‘Q’ 和 ‘.’ 分别代表皇后和空位。
    例.
    输入: n = 4
    输出: [
    [“.Q…”,  // 解法 1
    “…Q”,
    “Q…”,
    “…Q.”],

    [“…Q.”,  // 解法 2
    “Q…”,
    “…Q”,
    “.Q…”]
    ]
    """

    def __init__(self, n):
        self.queens = [-1] * n
        self.rows = ["."] * n
        self.cols = set()
        self.dia1 = set()
        self.dia2 = set()
        self.path = []
        self.n = n

    def backtrack(self, row, n):
        if row == n:
            arr = []
            for j in self.queens:
                self.rows[j] = "Q"
                arr.append("".join(c for c in self.rows))
                self.rows[j] = "."
            self.path.append(arr[:])

        for i in range(n):
            if i in self.cols or row - i in self.dia1 or row + i in self.dia2:
                continue
            self.queens[row] = i
            self.cols.add(i)
            self.dia1.add(row - i)
            self.dia2.add(row + i)
            self.backtrack(row + 1, n)
            self.queens[row] = -1
            self.cols.remove(i)
            self.dia1.remove(row - i)
            self.dia2.remove(row + i)

    def nq(self):
        self.backtrack(0, self.n)
        return self.path


def search_in_rotated_sorted_array(nums, target):
    right = len(nums) - 1
    left = 0
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] >= nums[left]:
            if nums[mid] > target >= nums[left]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[right] >= target > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


def find_minimum_in_rotated_sorted_array(nums):
    min_value = 1000000
    right = len(nums) - 1
    left = 0
    while left <= right:
        mid = (left + right) // 2
        # min_value=min(min_value,nums[mid])
        # if nums[mid] >= nums[left]:
        #     if nums[mid] > min_value >= nums[left]:
        #         right = mid - 1
        #     else:
        #         left = mid + 1
        # else:
        #     if nums[right] >= min_value > nums[mid]:
        #         left = mid + 1
        #     else:
        #         right = mid - 1
        if nums[mid] >= nums[left]:
            min_value = min(min_value, nums[left])
            left = mid + 1
        else:
            min_value = min(min_value, nums[mid])
            right = mid - 1
    return min_value


def largest_rectangle_in_histogram(heights):
    pre_stack = []
    pre_index = []
    for i in range(len(heights)):
        while len(pre_stack) and heights[pre_stack[-1]] >= heights[i]:
            pre_stack.pop()
        pre_index.append(pre_stack[-1] + 1 if len(pre_stack) else 0)
        pre_stack.append(i)
    last_stack = []
    last_index = []
    for i in range(len(heights) - 1, -1, -1):
        while len(last_stack) and heights[last_stack[-1]] >= heights[i]:
            last_stack.pop()
        last_index.append(last_stack[-1] - 1 if len(last_stack) else len(heights) - 1)
        last_stack.append(i)

    last_index = last_index[::-1]
    max_rectangle = 0
    for i in range(len(heights)):
        max_rectangle = max(max_rectangle, (last_index[i] - pre_index[i] + 1) * heights[i])
    return max_rectangle


def daily_temperatures(nums):
    stack = []
    temperatures = [0] * len(nums)
    for i in range(len(nums)):
        while len(stack) and nums[stack[-1]] < nums[i]:
            pre_index = stack.pop()
            temperatures[pre_index] = i - pre_index
        stack.append(i)
    return temperatures


def decode_string(s):
    stack = []
    current_char = ""
    current_num = 0
    for c in s:
        if c.isdigit():
            current_num = current_num * 10 + int(c)
        elif c == "[":
            stack.append((current_char, current_num))
            current_char = ""
            current_num = 0
        elif c == "]":
            last_char, repeat_times = stack.pop()
            current_char = last_char + current_char * repeat_times
        else:
            current_char += c
    return current_char


def valid_parentheses(s):
    stack = []
    dict = {"{": "}", "(": ")", "[": "]", "}": "", ")": "", "]": ""}
    for c in s:
        if len(stack) and dict[stack[-1]] == c:
            stack.pop()
        else:
            stack.append(c)
    if len(stack):
        return False
    else:
        return True


def median_of_two_sorted_arrays(nums1, nums2):
    if len(nums1) > len(nums2):
        return median_of_two_sorted_arrays(nums2, nums1)
    m = len(nums1)
    n = len(nums2)
    left = 0
    right = m
    infinty = 2 ** 40
    median1 = 0
    median2 = 0
    while left <= right:
        i = (left + right) // 2
        j = (m + n + 1) // 2 - i
        im_1 = -infinty if i == 0 else nums1[i - 1]
        im = infinty if i == m else nums1[i]
        jm_1 = -infinty if j == 0 else nums2[j - 1]
        jm = infinty if j == n else nums2[j]
        if im_1 <= jm:
            left = i + 1
            median1 = max(im_1, jm_1)
            median2 = min(im, jm)
        else:
            right = i - 1
    return median1 if (m + n) % 2 else (median1 + median2) / 2


def find_first(nums, target):
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


def find_last(nums, target):
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid - 1
    return left


def find_first_and_last_position_of_element_in_sorted_array(nums, target):
    first = find_first(nums, target)
    last = find_last(nums, target)
    if first < len(nums) and nums[first] == target:
        return [first, last - 1]
    else:
        return [-1, -1]


def search_a_2D_matrix(matrix, target):
    left = 0
    right = len(matrix) - 1
    while left <= right:
        mid = (left + right) // 2
        if matrix[mid][0] == target:
            return True
        elif matrix[mid][0] > target:
            right = mid - 1
        else:
            left = mid + 1
    index = left - 1
    if index < 0 or index > right:
        return False
    newmatrix = matrix[index]
    left = 0
    right = len(newmatrix) - 1
    while left <= right:
        mid = (left + right) // 2
        if newmatrix[mid] == target:
            return True
        elif newmatrix[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return False


def perfect_squares(n):
    """
    完全平方数
    找出最小的完全平方数的数量,使完全平方数之和恰好等于n
    例.
    输入: n = 12
    输出: 3
    解释: 12 = 4 + 4 + 4

    输入: n = 13
    输出: 2
    解释: 13 = 4 + 9
    状态转移方程：dp[i]=min(dp[i],dp[i-j**2])+1
    """
    dp = [2 ** 40] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        for j in range(1, int(math.sqrt(i)) + 1):
            dp[i] = min(dp[i], dp[i - j ** 2] + 1)
    return dp[n]


def house_robber(nums):
    """
    “打家劫舍”是 LeetCode 上一个经典的动态规划问题。题目要求你在一条街道上的一排房屋中偷窃最大金额,但不能连续偷窃相邻的房屋。
    每间房屋内都藏有一定的现金,你的目标是在不触动警报的情况下,尽可能多地偷窃现金。
    例.
    输入: nums = [1, 2, 3, 1]
    输出: 4
    解释: 偷窃第 1 个房屋 (金额 = 1) 和第 3 个房屋 (金额 = 3),总金额 = 1 + 3 = 4。

    输入: nums = [2, 7, 9, 3, 1]
    输出: 12
    解释: 偷窃第 1 个房屋 (金额 = 2), 第 3 个房屋 (金额 = 9) 和第 5 个房屋 (金额 = 1),总金额 = 2 + 9 + 1 = 12。
    """
    n = len(nums)
    dp = [0] * (n)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
    return dp[n - 1]


def pascal_triangle(n):
    """
    “杨辉三角”是 LeetCode 上一个经典的数组生成问题。题目要求生成杨辉三角的前 numRows 行。每一行的元素可以通过前一行的元素来生成。
    例.
    输入n=3
    [  [1]  ]
    [ [1 1] ]
    [[1 2 1]]
    """
    ans = [[1]]
    for i in range(1, n):
        tmp = []
        for j in range(i + 1):
            m = 0 if j == 0 else ans[i - 1][j - 1]
            n = 0 if j == i else ans[i - 1][j]
            tmp.append(m + n)
        ans.append(tmp)
    return ans


def climbing_stairs(n):
    """
    “爬楼梯”是 LeetCode 上一个经典的动态规划问题。题目要求计算出到达第 n 级台阶的方法总数,每次可以爬 1 级或 2 级台阶。
    状态转移方程:dp[i]=dp[i−1]+dp[i−2]
    例.
    dp[2]=1+1 or 2 两种
    """
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def partition_labels(s):
    """
    “划分字母区间”是 LeetCode 上一个经典的字符串处理问题。题目要求将字符串划分成一些子串,使得每个子串中的所有字符都在该子串中出现,并且这些子串的长度之和最小。每个子串中的字符在该子串中必须是唯一的。
    例.
    输入: s = "ababcbacadefegdehijhklij" 输出: [9, 14, 27] 解释: 划分结果为 “ababcbaca”, “defegde”, “hijhklij”,这三个子串分别在索引 9, 14, 27 结束。
    """
    hash_map = {}
    start = 0
    end = 0
    ans = []
    for i in range(len(s)):
        hash_map[s[i]] = i
    for i in range(len(s)):
        end = max(end, hash_map[s[i]])
        if end == i:
            ans.append(end - start + 1)
            start = i + 1
    return ans


def word_break(s, word_dict):
    """
    动态规划
    单词是否被列表中的单词拆分
    """
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    start = 0
    for i in range(1, n + 1):
        # for j in range(i):
        #     if dp[j] and s[j:i] in word_dict:
        #         dp[i] = True
        #         break
        if s[start:i] in word_dict and dp[start]:
            dp[i] = True
            start = i
    return dp[n]


def coin_change(amount, coins):
    """
    amount是否可以刚好用零钱兑换,如果不可以返回-1,如果可以返回最少需要几张零钱,每个面额可以重复使用
    例 amount=10 coins=[2,3,4]
    """

    def coin_change_rec(rem, memo):
        # 剩余的钱进行零钱兑换,是少还是有剩余
        if rem == 0:
            return 0
        if rem < 0:
            return -1
        if rem in memo:
            return memo[rem]
        mini = float('inf')
        for coin in coins:
            res = coin_change_rec(rem - coin, memo)
            if mini > res >= 0:
                # 如果返回值有效,记录该数额的最少找零钱个数
                mini = res + 1
        memo[rem] = -1 if mini == float('inf') else mini
        return memo[rem]

    memo = {}
    coin_change_rec(amount, memo)
    return memo[amount]


def longest_increasing_subsequence(nums):
    """
    最长递增子序列
    例 [5, 1, 2, 7, 8, 9, 2] 最长子序列=5 [1,2,7,8,9]
    """
    tails = []
    for num in nums:
        left, right = 0, len(tails) - 1
        while left <= right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left += 1
            else:
                right -= 1
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    print(tails)
    return len(tails)


def maximum_product_subarray(nums):
    """
    动态规划
    存在一个数组,计算最大的乘积
    例 [1,-5,2,6,-1] 最大乘积=60
    """
    if not nums:
        return 0
    max_product = nums[0]
    min_product = nums[0]
    result = nums[0]
    for i in range(1, len(nums)):
        if nums[i] < 0:
            tmp = max_product
            max_product = min_product
            min_product = tmp
        max_product = max(nums[i], max_product * nums[i])
        min_product = min(nums[i], min_product * nums[i])
        result = max(result, max_product)
    return result


def can_partition(nums):
    """
    动态规划
    判断列表是否存在一个子集,使得子集的和等于总和的一半
    例.
    [2,3,1,4] 存在,2+3=5 1+4=5
    [2,3,1] 存在 2+1=3
    [2,3] 不存在
    """
    total_nums = sum(nums)
    if total_nums % 2:
        return False
    target = total_nums // 2
    # 查找是否存在子集和等于target
    dp = [False] * (target + 1)
    dp[0] = True
    # [2,3,1,4]
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j - num]
    return dp[target]


def longest_valid_parentheses(s):
    """
    最长有效括号
    动态规划
    找到()))(()()中的最长有效括号
    ()()最长有效括号=2
    (())最长有效括号=4
    """
    n = len(s)
    if not n:
        return 0
    dp = [0] * (n)
    max_length = 0
    for i in range(1, n):
        if s[i] == ")":
            if s[i - 1] == "(":
                dp[i] = (dp[i - 2] if i - 2 >= 0 else 0) + 2
            elif i - dp[i - 1] > 0 and s[i - dp[i - 1] - 1] == "(":
                dp[i] = dp[i - 1] + 2 + dp[i - dp[i - 1] - 2]
            max_length = max(max_length, dp[i])
    return max_length


def jump_game(nums):
    """
    “跳跃游戏”是 LeetCode 上一个经典的数组问题。题目要求判断你是否能够从数组的起始位置跳到数组的末尾。每次跳跃的最大距离由当前元素决定。
    例.
    输入: nums = [2, 3, 1, 1, 4]
    输出: true
    解释: 从位置 0 跳到位置 1,再从位置 1 跳到位置 4。

    输入: nums = [3, 2, 1, 0, 4]
    输出: false
    解释: 无论怎样,总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ,所以永远不可能到达最后一个位置。
    """
    end = 0
    for i in range(len(nums)):
        if end >= i:
            end = max(end, i + nums[i])
            if end >= len(nums) - 1:
                return True
    return False


def jump_game2(nums):
    """
    “跳跃游戏 II”是 LeetCode 上一个经典的数组问题。题目要求在给定的非负整数数组 nums 中,找到从数组的起始位置到达数组末尾所需的最少跳跃次数。每次跳跃的最大距离由当前元素决定。
    例.
    输入: nums = [2, 3, 1, 1, 4]
    输出: 2
    解释: 跳到最后一个位置的最小跳跃次数是 2。从下标 0 跳到下标 1,再从下标 1 跳到下标 4。

    输入: nums = [2, 3, 0, 1, 4]
    输出: 2
    """
    end = 0
    times = 0
    max_end = 0
    for i in range(len(nums) - 1):
        max_end = max(max_end, i + nums[i])
        if end == i:
            end = max_end
            times += 1

    return times


def best_time_to_buy_and_sell_stock(nums):
    """
    “买卖股票的最佳时机”是 LeetCode 上一个经典的数组问题。题目要求在给定的价格数组中找到最佳的买入和卖出时机,以获得最大的利润。这里我们假设只能进行一次买卖操作。
    例.
    输入: prices = [7, 1, 5, 3, 6, 4]
    输出: 5
    解释: 在第 2 天（股票价格 = 1）买入,在第 5 天（股票价格 = 6）卖出,利润 = 6 - 1 = 5 。注意不能在买入之前卖出股票。

    输入: prices = [7, 6, 4, 3, 1]
    输出: 0
    解释: 在这种情况下,没有交易完成,所以最大利润为 0。
    """
    value = 0
    max_value = 0
    for i in range(1, len(nums)):
        sub = nums[i] - nums[i - 1]
        value = max(0, value + sub)
        max_value = max(max_value, value)
    return max_value


def unique_path(m, n):
    """
    动态规划
    不同路径和
    算从网格左上角到网格右下角经过的总共路径条数
    """
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]


def min_path_sum(grid):
    """
    动态规划
    计算最小路径和
    存在一个网格,计算从网格左上角到网格右下角经过的最小路径和
    """
    m = len(grid)
    n = len(grid[0])
    # 记录到m、n位置的最小路径和
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    # 不需要动态规划的部分可以直接计算获得
    for i in range(1, m):
        dp[i][0] = grid[i][0] + dp[i - 1][0]
    for j in range(1, n):
        dp[0][j] = grid[0][j] + dp[0][j - 1]
    for row in range(1, m):
        for col in range(1, n):
            dp[row][col] = min(dp[row][col - 1], dp[row - 1][col]) + grid[row][col]
    return dp[m - 1][n - 1]


def longest_palindrome(s):
    """
    找到字符串中的最长回文子串
    """
    m = len(s)
    if not m:
        return ""
    dp = [[True] * m for _ in range(m)]
    max_length = 1
    start = 0
    for i in range(m - 1, -1, -1):
        for j in range(i + 1, m):
            dp[i][j] = s[i] == s[j] and dp[i + 1][j - 1]
            if dp[i][j] and j - i + 1 > max_length:
                max_length = j - i + 1
                start = i
    return s[start:start + max_length]


def longest_common_subsequence(s1, s2):
    """
    您提到的是 LeetCode 上的“最长公共子序列”（Longest Common Subsequence, LCS）问题。这是一个经典的动态规划问题。题目描述如下：
    给定两个字符串 text1 和 text2，返回它们的最长公共子序列的长度。如果不存在公共子序列，则返回 0。
    """
    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def island_nums(grid):
    """
    您提到的是 LeetCode 上的“岛屿数量”（Number of Islands）问题。这是一道经典的深度优先搜索（DFS）或广度优先搜索（BFS）问题。题目描述如下：
    给定一个由 '1'（陆地）和 '0'（水）组成的二维网格，计算其中有多少个岛屿。岛屿是由水平或垂直相邻的陆地连接而成的，你可以假设网格的四个边界都被水包围。
    """

    def dfs(grid, r, c):
        # 判断边界
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
            return
        # 如果是海则停止dfs
        if grid[r][c] == 0:
            return
        # 如果是岛屿则使用海水淹没
        grid[r][c] = 0
        dfs(grid, r - 1, c)
        dfs(grid, r + 1, c)
        dfs(grid, r, c - 1)
        dfs(grid, r, c + 1)

    cnt = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                cnt += 1
                dfs(grid, r, c)
    return cnt


def oranges_rotting(grid):
    """
    您提到的是 LeetCode 上的“腐烂的橘子”（Rotting Oranges）问题。这是一道经典的广度优先搜索（BFS）问题。题目描述如下：
    在一个 m x n 的网格中，每个单元格可能有以下三种状态之一：
    0 表示该单元格为空。
    1 表示该单元格有一个新鲜的橘子。
    2 表示该单元格有一个腐烂的橘子。
    每分钟，任何与腐烂的橘子相邻的新鲜橘子都会腐烂。返回直到没有任何新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1。
    """
    qe = collections.deque()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # 记录腐烂橘子的位置
            if grid[i][j] == 2:
                qe.append((i, j, 0))
    time = 0
    while qe:
        r, c, time = qe.popleft()
        # 开始感染周围橘子
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < len(grid) and 0 <= cc < len(grid[0]):
                # 如果是新鲜橘子则被感染，感染的橘子放进队列
                if grid[rr][cc] == 1:
                    qe.append((rr, cc, time + 1))
                    grid[rr][cc] = 2
    cnt = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # 查看是否还有没被感染的新鲜橘子
            if grid[i][j] == 1:
                cnt += 1
    return -1 if cnt else time


def can_finish(numCourses, prerequisites):
    """
    你总共要完成 numCourses 门课，这些课程的先修关系会以一个数组 prerequisites 的形式给出，其中 prerequisites[i] = [ai, bi]
    表示如果你想要学习课程 ai，那么你必须先学习课程 bi。判断是否可以完成所有课程的学习。
    """
    degrees = [0] * numCourses
    dic = defaultdict(list)
    for info in prerequisites:
        # 学会info[1],就能学习新的课info[0]
        dic[info[1]].append(info[0])
        # 学习info[0]需要先学会的课的数目
        degrees[info[0]] += 1
    dq = collections.deque()
    for i in range(len(degrees)):
        # 找到哪些课可以直接学，没有前序课程
        if degrees[i] == 0:
            dq.append(i)
    visited = 0
    while dq:
        degree = dq.popleft()
        visited += 1
        for d in dic[degree]:
            degrees[d] -= 1
            if degrees[d] == 0:
                dq.append(d)
    return visited == numCourses


class Trie:
    """
    前缀是否存在
    通过insert插入单词
    通过search查询单词是否存在
    通过find_prefix查询是否有该前缀的单词存在
    例.
    insert("word")
    search("word")存在
    search("wor")不存在 wor不是一个单词
    find_prefix("wor")存在 前缀是wor的单词word
    """

    def __init__(self):
        self.children = [None] * 26
        # 保存完一个单词后设置为True
        self.is_end = False

    def insert(self, word):
        node = self
        for c in word:
            # 将一个小写字母转换为从0开始的索引
            ch = int(ord(c) - ord("a"))
            # 如果第ch个索引是None，则赋值一个新的Trie
            if not node.children[ch]:
                node.children[ch] = Trie()
            # 如果单词的第i个字符存在，则跳转到第i+1
            node = node.children[ch]

        self.is_end = True

    def search(self, prefix):
        node = self
        for c in prefix:
            ch = int(ord(c) - ord("a"))
            if not node.children[ch]:
                return False, node
            node = node.children[ch]
        return True, node

    def find_prefix(self, prefix):
        result, node = self.search(prefix)
        if result:
            return True
        return False


class MedianInDataStream():
    """
    堆
    搜索数据流中的中位数
    例.
    [2,1,3] 中位数2
    [6,5,4,7] 中位数5.5
    """

    def __init__(self):
        self.min_heap = []
        self.max_heap = []

    def insert(self, num):
        heapq.heappush(self.max_heap, -num)
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def search(self):
        if len(self.min_heap) < len(self.max_heap):
            return -self.max_heap[0]
        else:
            return (self.min_heap[0] - self.max_heap[0]) / 2


def min_distance(s1, s2):
    """
    “编辑距离”是 LeetCode 上的一个经典问题,通常对应的是第 72 题 “Edit Distance”。
    这个问题要求计算将一个字符串转换成另一个字符串所需的最少单字符编辑操作次数（插入、删除或替换）。

    """
    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1
    return dp[m][n]

def top_k(nums,k):
    heap_lst=[]
    hash_map=Counter(nums)
    for key,freq in hash_map.items():
        heapq.heappush(heap_lst,(freq,key))
        if len(heap_lst)>k:
            heapq.heappop(heap_lst)
    return [key for freq,key in heap_lst]