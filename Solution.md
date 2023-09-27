
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

46. Permutations
###### ```java
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
