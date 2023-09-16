
###### 40. Combination Sum II
```java
public List<List<Integer>> combinationSum2(int[] candidates, int target){
   List<List<Integer>> list = new LinkedList<List<Integer>>();
   Arrays.sort(candidates);
   backtrack(list, new ArrayList<Integer>(), candidates, target, 0);
   return list;
}
private void backtrack(List<List<Integer>> list, List<Integer> tempList, int[] cand, int remain, int start){
   if(remain < 0) return; /** no solution */
   else if(remain == 0) list.add(new ArrayList<>(tempList));
   else{
      for (int i = start; i < cand.length; i++) {
         if(i > start && cand[i] == cand[i-1]) continue; /** skip duplicates */
         tempList.add(cand[i]);
         backtrack(list, tempList, cand, remain - cand[i], i+1);
         tempList.remove(tempList.size() - 1);
      }
   }
}
```
- This code uses DFS to check every possible conbination and it goes through whether a number is picked or not.<br>
- Sorting array is to prevent dupicate combination, for array like [10,1,2,7,6,1,5], it will generate [1,7] and [7,1] if not sorted.<br>
- Line 15: For array like [1,1,2,5,6,7,10], the first "for" loop before any recursion will trigger continue if any duplicates detected.<br>
- However, when the function keeps calling it self, "i" will be updated with new "start", so it won't trigger continue since "i" equal<br>
to "start". Therefore, "tempList" could be [1,1] and now the duplicates are allowed. 
