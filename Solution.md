
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
we may get a smaller element somewhere ahead which is still greater than previous and picking that would be optimal. So we try both options.
