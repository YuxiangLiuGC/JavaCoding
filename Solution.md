
###### 40. Combination Sum II
```java
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> list = new ArrayList<>();
        Arrays.sort(candidates);
        helper(list,new ArrayList<Integer>(), candidates, target, 0);
        return list;
    }
    public void helper(List<List<Integer>> list, List<Integer> tempList, int[] cand, int remain, int start){
        if(remain<0) return; //no solution
        else if(remain==0){
            list.add(new ArrayList<>(tempList));
        }else{
            for(int i=start; i<cand.length; i++){
                if(i>start && cand[i]==cand[i-1]) continue; 
                tempList.add(cand[i]);
                helper(list, tempList, cand, remain-cand[i], i+1);
                tempList.remove(tempList.size()-1);
            }
        }
    }
}
```
- This code uses DFS to check every possible conbination and it goes through whether a number is picked or not.<br>
- Sorting array is to prevent dupicate combination, for array like [10,1,2,7,6,1,5], it will generate [1,7] and [7,1] if not sorted.<br>
- **if(i > start && cand[i] == cand[i-1])**: For array like [1,1,2,5,6,7,10], in the first "for" loop, it will trigger "continue" if any duplicates detected.<br>
- However, every time the function keeps calling it self, "i" will be updated with new "start", so it won't check duplicates when "i" equal<br>
to "start". Therefore, "tempList" could be [1,1] and now the duplicates are allowed.

###### 78. Subsets
```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        helper(nums, list, new ArrayList<Integer>(), 0);
        return list;
    }
    private void helper(int[] nums, List<List<Integer>> list, List<Integer> temp, int start){
        list.add(new ArrayList<>(temp));
        
        for(int i=start; i<nums.length; i++){
            temp.add(nums[i]);
            helper(nums, list, temp, i+1);
            temp.remove(temp.size()-1);
        }
    }
}
```
- Time: O(n*n^2), where n is the length of input array
- 
###### 90. Subsets II
```java
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        Arrays.sort(nums);
        helper(nums, list, new ArrayList<Integer>(), 0);
        return list;
    }
    private void helper(int[] nums, List<List<Integer>> list, List<Integer> temp, int start){
        list.add(new ArrayList<>(temp));
        
        for(int i=start; i<nums.length; i++){
            if(i>start && nums[i]==nums[i-1]){
                continue;
            }
            temp.add(nums[i]);
            helper(nums, list, temp, i+1);
            temp.remove(temp.size()-1);
        }
    }
}
```

###### 39. Combination Sum
```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> list = new ArrayList<>();
        Arrays.sort(candidates);
        helper(candidates, list, new ArrayList<Integer>(), target, 0);
        return list;
    }
    private void helper(int[] candidates, List<List<Integer>> list, List<Integer> temp, int remain, int start){
        if(remain<0) return;
        else if(remain==0){
            list.add(new ArrayList<Integer>(temp));
        }else{
            for(int i=start; i<candidates.length; i++){
                temp.add(candidates[i]);
                helper(candidates, list, temp, remain-candidates[i], i);
                temp.remove(temp.size()-1);
            }
        }
    }
}
```

###### 46. Permutations
```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        helper(list, nums, new ArrayList<Integer>());
        return list;
    }
    private void helper(List<List<Integer>> list, int[] nums, List<Integer> temp){
        if(temp.size()==nums.length){
            list.add(new ArrayList<Integer>(temp));
        }else{
            for(int i=0; i<nums.length; i++){
                if(temp.contains(nums[i])) continue;
                temp.add(nums[i]);
                helper(list, nums, temp);
                temp.remove(temp.size()-1);
            }
        }
    }
}
```
- Time complexity: O(n*n!)
- Given a set of length n, the number of permutations is n factorial. There are n options for the first number, n - 1 for the second, and so on.

###### 131. Palindrome Partitioning
```java
class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> list = new ArrayList<>();
        helper(list, s, new ArrayList<String>(), 0);
        return list;
    }
    private void helper(List<List<String>> list, String s, List<String> temp, int start){
        if(start==s.length()){
            list.add(new ArrayList<String>(temp));
        }else{
            for(int i=start; i<s.length(); i++){
                if(isPalindrome(s, start, i)){
                    temp.add(s.substring(start, i+1));
                    helper(list, s, temp, i+1);
                    temp.remove(temp.size()-1);
                }
            }
        }
    }
    private boolean isPalindrome(String s, int left, int right){
        while(left<right){
            if(s.charAt(left)!=s.charAt(right)){
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}
```
- Time Complexity : O(N*2^N), where N is the length of string s.
- there could be 2^N possible substrings in the worst case. For each substring, it takes O(N) time to generate the substring and determine if it is a palindrome or not.

###### 17. Letter Combinations of a Phone Number
```java
class Solution {
    public List<String> letterCombinations(String digits) {
        if(digits.isEmpty()) return new ArrayList<>();
        String[] map = {"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        List<String> res = new ArrayList<>();
        helper("", res, digits, map);
        return res;
    }
    private void helper(String combination, List<String> res, String digits, String[] map){
        if(digits.isEmpty()){
            res.add(combination);
        }else{
            String letters = map[digits.charAt(0)-'2']; // '2'-'2' = 0
            for(char letter: letters.toCharArray()){
                helper(combination+letter, res, digits.substring(1), map);
            }
        }
    }
}
```
- Time complexity: ( O(4^n) ), where ( n ) is the length of the input string. In the worst case, each digit can represent 4 letters, so there will be 4 recursive calls for each digit.
  
###### 51. N-Queens
```java
class Solution {
        Set<Integer> colCheck = new HashSet<>();
        Set<Integer> diag1 = new HashSet<>();
        Set<Integer> diag2 = new HashSet<>();
    
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        
        helper(res, new ArrayList<String>(), 0, n);
        return res;
    }
    private void helper(List<List<String>> res, List<String> temp, int row, int n){
        if(row==n){
            res.add(new ArrayList<>(temp));
        }else{
            for(int col=0; col<n; col++){
                if(colCheck.contains(col) || diag1.contains(row+col) || diag2.contains(row-col)){
                    continue;
                }
                char[] arr = new char[n];
                Arrays.fill(arr,'.');
                arr[col] = 'Q';
                String s = new String(arr);
                
                temp.add(s);
                colCheck.add(col);
                diag1.add(row+col);
                diag2.add(row-col);
                
                helper(res, temp, row+1, n);
                
                temp.remove(s);
                colCheck.remove(col);
                diag1.remove(row+col);
                diag2.remove(row-col);
            }
        }
    }
}
```
- Time complexity: O(N!)
  
###### 1436. Destination City
```java
//Since there's only one destination city, if a city isn't a source city, it has to be the destination.
public String destCity(List<List<String>> paths) {
        Set<String> cities = new HashSet<>(); 
        for (List<String> path : paths) {
            cities.add(path.get(0)); 
        }
        
        for (List<String> path : paths) {
            String dest = path.get(1); 
            if (!cities.contains(dest)) {
                return dest; 
            }
        }
        return "";
    }
```

###### 300. Longest Increasing Subsequence
- Only comparing the prev and curr value doesn't work, because you may pick numbers that are not next to the previous ones.<br>
Additionally, if the current element is greater than the previous picked element, then we can either pick it or don't pick it because<br>
we may get a smaller element somewhere ahead which is still greater than previous and picking that would be optimal.<br>
- So we use dynamic programming...For each slot in dp array, we are trying to find out the longest subsequence by far "i" if including the<br>
curr "i"(The reason of plus 1). We are not sure which one is the optimal, so we use "j" to traverse to find out
- Don't forget to go through each element in dp to find out the longest subsequence
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        
        for(int i=1; i<nums.length; i++){
            for(int j=0; j<i; j++){
                if(nums[i]>nums[j]){
                    dp[i] = Math.max(dp[i], dp[j]+1);
                }
            }
        }
        
        int longest = 1;
        for(int num: dp){
            longest = Math.max(longest, num);
        }
        return longest;
    }
}
```
###### 79. Word Search
```java
class Solution {
    boolean[][] visited; //The same letter cell may not be used more than once
    public boolean exist(char[][] board, String word) {
        visited = new boolean[board.length][board[0].length];
        for(int row=0; row<board.length; row++){
            for(int col=0; col<board[0].length; col++){
                if(helper(board, row, col, word, 0)){
                    return true;
                }
            }
        }
        return false;
    }
    private boolean helper(char[][] board, int row, int col, String word, int index){
        if(index==word.length()){
            return true;
        }
        if(row<0 || row>=board.length || col<0 || col>=board[0].length || word.charAt(index)!=board[row][col]
          || visited[row][col]){
            return false;
        }
        visited[row][col]=true;
        if(helper(board, row+1, col, word, index+1)||
            helper(board, row, col+1, word, index+1)||
            helper(board, row-1, col, word, index+1)||
             helper(board, row, col-1, word, index+1)){
            return true;
        }
        visited[row][col]=false;
        return false;
    }
}
```
- Time Complexity: O(N*3^L), where N is the number of cells in the board and L is the length of the word to be matched.
- Initially we could have at most 4 directions to explore, but further the choices are reduced into 3 (since we won't go back to where we come from).

###### 416. Partition Equal Subset Sum
Approach 1: DFS (Time Limit Exceeded)
```java
class Solution {
    public boolean canPartition(int[] nums) {
        int sum=0;
        for(int num: nums){
            sum+=num;
        }
        if(sum%2!=0) return false;
        sum = sum/2;
        return helper(nums, 0, sum);
    }
    private Boolean helper(int[] nums, int pos, int sum){
        if(sum==0) return true;
        if(pos>=nums.length || sum<0) return false;
        Boolean result = helper(nums, pos+1, sum-nums[pos]) || helper(nums, pos+1, sum); 
        return result;
    }
}
```
Approach 2: DP Memoization
```java
class Solution {
    Boolean[][] arr;
    public boolean canPartition(int[] nums) {
        int sum=0;
        for(int num: nums){
            sum+=num;
        }
        if(sum%2!=0) return false;
        sum = sum/2;
        arr = new Boolean[nums.length+1][sum+1];
        return helper(nums, 0, sum);
    }
    private Boolean helper(int[] nums, int pos, int sum){
        if(sum==0) return true;
        if(pos>=nums.length || sum<0) return false;
        
        if(arr[pos][sum]!=null) return arr[pos][sum];
        Boolean result = helper(nums, pos+1, sum-nums[pos]) || helper(nums, pos+1, sum); 
        arr[pos][sum] = result;
        return result;
    }
}
```

###### 19. Remove Nth Node From End of List
```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode fast=head, slow=head;
        
        for(int i=0; i<n; i++){
            fast = fast.next;
        }
        if(fast==null) return head.next;
        while(fast.next!=null){
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        
    return head;
    }
}
//1  ->  2   ->   3  ->   4  ->   5  ->  null   (n=2)
//               fast                           (The for loop will result in)
//slow                                          (Where slow starts)
//               slow            fast           (The while loop will result in)

//Why do we need if(fast==null) return head.next? 
//1  ->   2  ->   null     (n=2)
//                fast
//if not fast.next!=null will cause error bc fast is null already
```

###### 14. Longest Common Prefix
```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        int shortest=Integer.MAX_VALUE;
        for(int i=0; i<strs.length; i++){
            if(strs[i].length()<shortest){
                shortest = strs[i].length();
            }
        }
        int i=0;
        for(i=0; i<shortest; i++){
            char c = strs[0].charAt(i);
            for(int j=1; j<strs.length; j++){
                if(strs[j].charAt(i) != c){
                    return strs[0].substring(0, i); // cannot use break here bc it only stops the inner for loop
                }
            }
        }
        return strs[0].substring(0, i);
    }
}
```

###### 394. Decode String
```java
class Solution {
    int i = 0;
    public String decodeString(String s) {
        StringBuilder sb = new StringBuilder();
        int count = 0;
        
        while(i<s.length()){
            char c = s.charAt(i); // Assume s is valid and the first character is number in the first place
            i++; // we need to plus 1 before we go to the subproblem
            
            if(c=='['){
                String temp = decodeString(s); // do subproblem
                for(int j=0; j<count; j++){
                    sb.append(temp);
                }
                count=0; // For example like "3[a]2[bc]", there is a another '[' behind
                
            }else if(c==']'){
                break;
            }else if(Character.isAlphabetic(c)){
                sb.append(c);
            }else{
                count = count * 10 + c - '0';// convert the character to integer
            }
        }
        return sb.toString();
    }
}
```
###### 438. Find All Anagrams in a String
```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        if(s.length()<p.length()) return res;
             
        Map<Character, Integer> sMap = new HashMap<>();
        Map<Character, Integer> pMap = new HashMap<>();
        
        for(int i=0; i<p.length(); i++){
            sMap.put(s.charAt(i), sMap.getOrDefault(s.charAt(i),0)+1);
            pMap.put(p.charAt(i), pMap.getOrDefault(p.charAt(i),0)+1);
        }
        if(sMap.equals(pMap)){
            res.add(0);
        }
        int left=0, right=p.length();
        while(right<s.length()){
            char addition = s.charAt(right);
            sMap.put(addition, sMap.getOrDefault(addition, 0)+1);
            
            char removal = s.charAt(left);
            sMap.put(removal, sMap.get(removal)-1);
            left++;
            
            if(sMap.get(removal)==0){
                sMap.remove(removal);
            }
            if(sMap.equals(pMap)){
                res.add(left);
            }
            right++;
        }
        return res;
    }
}
```

###### 347. Top K Frequent Elements
```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int num: nums){
            map.put(num, map.getOrDefault(num,0)+1);
        }
        Queue<Integer> pq = new PriorityQueue<>((a,b)->map.get(b)-map.get(a));
        for(int num: map.keySet()){
            pq.add(num);
        }
        int[] res = new int[k];
        for(int i=0; i<k; i++){
            res[i] = pq.poll();
        }
        return res;
    }
}
```

###### 238. Product of Array Except Self
```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        int[] left = new int[nums.length];
        int[] right = new int[nums.length];

        left[0] = 1;
        int product = 1;
        for(int i=1; i<nums.length; i++){
            left[i] = nums[i-1]*product;
            product = left[i];
        }
        right[nums.length-1] = 1;
        product = 1;
        for(int i=nums.length-2; i>=0; i--){
            right[i] = nums[i+1]*product;
            product = right[i];
        }
        for(int i=0; i<nums.length; i++){
            res[i] = left[i]*right[i];
        }
        return res;
    }
}
```

###### 36. Valid Sudoku
```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        Set<Character>[] rows = new HashSet[9];
        Set<Character>[] cols = new HashSet[9];
        Set<Character>[] boxes = new HashSet[9];

        for(int i=0; i<9; i++){
            rows[i] = new HashSet<Character>();
            cols[i] = new HashSet<Character>();
            boxes[i] = new HashSet<Character>();
        }

        for(int i=0; i<board.length; i++){
            for(int j=0; j<board[0].length; j++){
                char target = board[i][j];
                if(target=='.') continue;
                if(rows[i].contains(target)){
                    return false;
                }
                rows[i].add(target);

                if(cols[j].contains(target)){
                    return false;
                }
                cols[j].add(target);
                //Calculate which grid it belongs to. Each row has 3 boxes so it times 3
                int index = (i/3)*3 + j/3;
                if(boxes[index].contains(target)){
                    return false;
                }
                boxes[index].add(target);
            }
        }
        return true;
    }
}
```

###### 271. Encode and Decode Strings
```java
public class Codec {

    // Encodes a list of strings to a single string.
    public String encode(List<String> strs) {
        StringBuilder sb = new StringBuilder();
        for(String s: strs){
            sb.append(s.replace("#", "##")).append(" # ");
        }
        return sb.toString();
    }

    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        List<String> res = new ArrayList<>();
        String[] arr = s.split(" # ", -1); // We need -1 bc empty string "" need to be kept
        for(int i=0; i<arr.length-1; i++){ // trailing empty string
            res.add(arr[i].replace("##", "#"));
        }
        return res;
    }
}
```

###### 128. Longest Consecutive Sequence
Approach 1: sorting
```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if(nums.length==0) return 0;
        Arrays.sort(nums);
        int longest=1, curr=1; // As least the longest and curr are 1

        // 3 cases after sorted: when nums[i] to nums[i-1] is "more than plus 1", "equal to plus 1" and "just equal"
        for(int i=1; i<nums.length; i++){ 
            if(nums[i]!=nums[i-1]){
                if(nums[i]==nums[i-1]+1){
                    curr++;
                }else{ // subsequence stoped
                    longest = Math.max(longest, curr);
                    curr = 1;
                }
            }
        }
        return Math.max(longest, curr); 
    }
}
```
Approach 2: HashSet
```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if(nums.length==0) return 0;
        Set<Integer> set = new HashSet<>();
        int longest = 1, count = 1;;

        for(int num: nums){
            set.add(num);
        }
        for(int num: nums){
            if(!set.contains(num-1)){
                int x = num;
                count = 1;
                while(set.contains(x+1)){
                    count++;
                    x++;
                }
            }
            longest = Math.max(longest, count);
        }
        return longest;
    }
}
```
###### 235. Lowest Common Ancestor of a Binary Search Tree
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null) return null;
        while(root!=null){
            if(p.val<root.val && q.val<root.val){
                root = root.left;
            }else if(p.val>root.val && q.val>root.val){
                root = root.right;
            }else{
                return root;
            }
        }
        return null;
    }
}
```

###### 102. Binary Tree Level Order Traversal
```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> levelOrder(TreeNode root) {
        if(root==null) return res;
        helper(root, 0);
        return res;
    }
    private void helper(TreeNode root, int level){
        if(root==null) return;
        if(res.size()==level){
            res.add(new ArrayList<>());
        }
        res.get(level).add(root.val);
        helper(root.left, level+1);
        helper(root.right, level+1);
    }
}
```
###### 199. Binary Tree Right Side View
```java
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> rightSideView(TreeNode root) {
        helper(root, 0);
        return res;
    }
    private void helper(TreeNode root, int level){
        if(root==null) return;
        if(res.size()==level){
            res.add(root.val);
        }
        helper(root.right, level+1);
        helper(root.left, level+1);
    }
}
```

###### 1448. Count Good Nodes in Binary Tree
```java
class Solution {
    int count=0;
    public int goodNodes(TreeNode root) {
        helper(root, Integer.MIN_VALUE);
        return count;
    }
    private void helper(TreeNode root, int greatest){
        if(root==null) return;
        if(root.val >= greatest) count++;
        greatest = Math.max(greatest,root.val);
        helper(root.left, greatest);
        helper(root.right, greatest);
    }
}
```

###### 98. Validate Binary Search Tree
```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return helper(root, null, null);
    }
    private Boolean helper(TreeNode root, Integer lowest, Integer highest){
        if(root==null) return true;
        if(lowest!=null && root.val<=lowest) return false;
        if(highest!=null && root.val>=highest) return false;
        return helper(root.left, lowest, root.val) && helper(root.right, root.val, highest);
    }
}
```

###### 230. Kth Smallest Element in a BST
```java
class Solution {
    List<Integer> list = new ArrayList<>();
    public int kthSmallest(TreeNode root, int k) {
        inOrder(root);
        return list.get(k-1);
    }
    private void inOrder(TreeNode root){
        if(root==null) return;
        inOrder(root.left);
        list.add(root.val);
        inOrder(root.right);
    }
}
```

###### 105. Construct Binary Tree from Preorder and Inorder Traversal
```java
class Solution {
    int i=0, p=0;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return helper(preorder, inorder, Integer.MIN_VALUE);
    }
    private TreeNode helper(int[] preorder, int[] inorder, int stop){
        if(p == preorder.length) return null;
        if(inorder[i] == stop){ // until we cover the last element to build the subtree and meet root
            i++;
            return null;
        }
        TreeNode root = new TreeNode(preorder[p]);
        p++;
        root.left = helper(preorder, inorder, root.val);//Keep passing new stop to subtree
        root.right = helper(preorder, inorder, stop);
        return root;
    }
}
// We don't know where the right subtree need to be split in the first place, so we pass in MIN_VALUE
// Keep partitioning until we reach the last node so that inorder[i] equals stop
```

###### 124. Binary Tree Maximum Path Sum
```java
class Solution {
    int max = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        helper(root);
        return max;
    }
    private int helper(TreeNode root){
        if(root==null) return 0;

        int left = Math.max(helper(root.left), 0);// Include path or not
        int right = Math.max(helper(root.right), 0);
        
        max = Math.max(max, root.val + left + right);

        return root.val + Math.max(left, right);
    }
}
// For every node, we try to include path from both left and right side so we won't
// miss any chance of finding the path from a subtree that has the largest value.
// When going up, we need to pick a path that is either left or right as the return value.
// If either the sum of left and right are neagtive, we rather not include either of them
```

###### 297. Serialize and Deserialize Binary Tree
```java
public class Codec {
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if(root==null) return "#";
        return root.val + "," + serialize(root.left) + "," + serialize(root.right);
    }
    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> queue = new LinkedList<>(Arrays.asList(data.split(",")));
        return helper(queue);
    }
    private TreeNode helper(Queue<String> queue){
        String s = queue.poll();
        if(s.equals("#")) return null; // Use equals to compare strings
        TreeNode root = new TreeNode(Integer.valueOf(s));
        root.left = helper(queue);
        root.right = helper(queue);
        return root;
    }
}
```

###### 208. Implement Trie (Prefix Tree)
```java
class TrieNode{
    Boolean isWord;
    TrieNode[] array;

    public TrieNode(){
        this.isWord = false;
        this.array = new TrieNode[26];
    }
}
class Trie {
    TrieNode root;
    public Trie() {
       root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode node = root;
        for(char c: word.toCharArray()){
            int i = c - 'a';
            if(node.array[i]==null){
                node.array[i] = new TrieNode();
            }
            node = node.array[i];
        }
        node.isWord = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for(char c: word.toCharArray()){
            int i = c - 'a';
            if(node.array[i]==null){
                return false;
            }
            node = node.array[i];
        }
        return node.isWord;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for(char c: prefix.toCharArray()){
            int i = c - 'a';
            if(node.array[i]==null){
                return false;
            }
            node = node.array[i];
        }
        return true;
    }
}
```
