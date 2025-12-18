from leetcode.algo import (max_in_sliding_window, nums_of_k, mainWindow, max_sub_string_without_cover, dot_without_self,
                           group_anagrams, longest_consecutive_sequence
, move0, max_water, collecting_rainwater, three_sum, lunzhuanshuzu, finding_anagrams, add, permuate, sum_of_two_numbers,
                           maximum_subarray_sum, merge_intervals, miss_frist, matrix_zero, rotate_matrix,
                           spiral_matrix, subset, phone_number_letter_combination, island_perimeter, CombinationSum,
                           GenerateParentheses, WordSearch, PalindromePartitioning, NQueen,
                           search_in_rotated_sorted_array,
                           find_minimum_in_rotated_sorted_array, largest_rectangle_in_histogram, daily_temperatures,
                           decode_string, valid_parentheses, median_of_two_sorted_arrays,
                           find_first_and_last_position_of_element_in_sorted_array,
                           search_a_2D_matrix, perfect_squares, house_robber, pascal_triangle, climbing_stairs,
                           word_break, coin_change, longest_increasing_subsequence, maximum_product_subarray,
                           can_partition, longest_valid_parentheses, partition_labels, jump_game, jump_game2,
                           best_time_to_buy_and_sell_stock, unique_path, min_path_sum, longest_palindrome,
                           longest_common_subsequence, island_nums,
                           oranges_rotting, can_finish,Trie,MedianInDataStream,min_distance,top_k)



if __name__ == "__main__":
    nums = [1, 2, 3, 0, 2, 3, -3]
    print(f"两数之和为target的存在{sum_of_two_numbers(nums, 3)}组")
    words = ["abc", "cba", "rf", "fr"]
    print(f"异位词分组结果为:{group_anagrams(words)}")
    print(f"最长连续序列长度为:{longest_consecutive_sequence(nums)}")
    print(f"交换0的结果为:{move0(nums)}")
    print(f"容器最多能盛的水为:{max_water(nums)}")
    print(f"3数之和结果为:{three_sum(nums, 6)}")
    print(f"接雨水结果为:{collecting_rainwater(nums)}")
    seq = "asajfdeaas"
    print(f"最大不重叠长度为:{max_sub_string_without_cover(seq)}")
    print(f"寻找异位词的结果为:{finding_anagrams(seq, 'as')}")
    print(f"和为k的子数组为:{nums_of_k(nums, 3)}")
    print(f"滑动窗口最大值为:{max_in_sliding_window(nums, 3)}")
    print(f"最小覆盖字串为:{mainWindow(seq, 'sd')}")
    print(f"最大子数组和为:{maximum_subarray_sum(nums)}")
    lists = [[1, 2], [1, 3], [3, 4], [2, 3], [6, 8], [7, 9]]
    print(f"合并区间为:{merge_intervals(lists)}")
    print(f"轮转数组为:{lunzhuanshuzu(nums, 3)}")
    print(f"除自身以外数组的乘积为:{dot_without_self(nums)}")
    print(f"缺失的第一个正数为:{miss_frist(nums)}")
    lists = [[1, 5, 1, 2, 5], [-1, 2, 0, 5, 6], [2, 0, 3, 0, 4], [6, -1, 0, 5, 4], [2, 2, 2, 0, 2]]
    print(f"矩阵置0为:{matrix_zero(lists)}")
    lists = [[1, 5, 1, 2, 5], [-1, 2, 0, 5, 6], [2, 0, 3, 0, 4], [6, -1, 0, 5, 4], [2, 2, 2, 0, 2]]
    print(f"旋转矩阵为:{rotate_matrix(lists)}")
    print(f"螺旋矩阵为:{spiral_matrix(lists)}")
    nums = [0, 1, 2]
    print(f"子集为:{subset(nums)}")
    print(f"全排列为:{permuate(nums, 0, len(nums))}")
    print(f"电话号码的字母组合为:{phone_number_letter_combination(0, [2, 3])}")
    lists = [[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
    print(f"岛屿周长为:{island_perimeter(lists)}")
    candidates = [1,2,3,5]
    target = 5
    c = CombinationSum(target, candidates)
    print(f"组合总和为:{c.get()}")
    gp = GenerateParentheses()
    print(f"生成括号为:{gp.gp(0, 0, 3)}")
    ws = WordSearch()
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    word = "aab"
    print(f"单词搜索:{ws.ws(board, word)}")
    pp = PalindromePartitioning()
    print(f"回文串搜索为:{pp.pp(word)}")
    nq = NQueen(4)
    print(f"N皇后为:{nq.nq()}")
    nums = [5, 6, 7, 9, 1, 2, 3]
    target = 2
    print("搜索旋转排序数组:", search_in_rotated_sorted_array(nums, target))
    print("最小值旋转排序数组:", find_minimum_in_rotated_sorted_array(nums))
    print("柱状图中最大的矩形:", largest_rectangle_in_histogram([2, 1, 5, 6, 2, 3]))
    print("每日温度:", daily_temperatures([2, 1, 2, 6, 2, 3]))
    s = "3[a]2[bc]"
    print("字符串解码:", decode_string(s))
    print("有效的括号:", valid_parentheses("()[{}]"))
    print("两个正序数组的中位数:", median_of_two_sorted_arrays([1, 2, 3, 4, 4], [6]))
    print("在排序数组中查找元素的第一个和最后一个位置:",
          find_first_and_last_position_of_element_in_sorted_array([1, 5, 5, 6, 6, 7, 7, 7, 8], 4))
    matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    print("搜索二维矩阵:", search_a_2D_matrix(matrix, 70))
    print("完全平方数为:", perfect_squares(13))
    print("打家劫舍为:", house_robber([2, 3, 4, 5, 3]))
    print("杨辉三角为:", pascal_triangle(10))
    print("爬楼梯为:", climbing_stairs(4))
    print("单词拆分为:", word_break("leetcode", ["leet", "code"]))
    print("找零钱:", coin_change(15, [3, 10, 5]))
    print("最长递增子序列:", longest_increasing_subsequence([10,9,2,5,3,7,101,18]))
    print("乘积最大子数组:", maximum_product_subarray([-2, 2, 3, 4]))
    print("分割等和子集:", can_partition([2,3,1,4]))
    print("最长有效括号为:", longest_valid_parentheses("(()))()(()"))
    print("划分字母区间:", partition_labels("ababcbacadefegdehijhklij"))
    print("跳跃游戏为:", jump_game([3, 2, 1, 0, 4]))
    print("跳跃游戏2为:", jump_game2([2, 3, 1, 1, 4]))
    print("最佳买卖股票时机为:", best_time_to_buy_and_sell_stock([7, 1, 5, 3, 6, 4]))
    print("不同路径为:", unique_path(3, 7))
    print("最小路径和为:", min_path_sum([
        [1, 3, 1],
        [1, 5, 1],
        [4, 2, 1]
    ]))
    print("最长回文子串为:", longest_palindrome("assad"))
    print("最长公共子序列为:", longest_common_subsequence("asds", "aasassda"))
    print("岛屿数量为:", island_nums([[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]]))
    print("腐烂橘子为:", oranges_rotting([[0, 0, 1, 0], [0, 1, 2, 0], [0, 0, 1, 0], [0, 1, 1, 0]]))
    print("课程表为:", can_finish(4, [
        [1, 0],
        [2, 0],
        [3, 1],
        [3, 2]
    ]))
    trie=Trie()
    trie.insert("words")
    print("前缀是否存在:",trie.find_prefix("word"))
    mids=MedianInDataStream()
    mids.insert(6)
    mids.insert(5)
    mids.insert(10)
    mids.insert(3)
    mids.insert(4)
    mids.insert(2)
    mids.insert(1)
    print("数据流的中位数为:",mids.search())
    print("编辑距离为:",min_distance("horsej","rse"))
    print("前k个高频元素:",top_k([2,2,2,1,1,3,3,3,3,4,4,5,6,6,6],1))