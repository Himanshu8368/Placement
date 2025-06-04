#include <vector>
#include <set>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        set<vector<int>>ans;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size();i++){
            if(i!=0 && nums[i]==nums[i-1]) continue;
            int target = -(nums[i]);
            int low =i+1; int high = nums.size()-1;
            while(low<high){
                int sum = nums[low]+nums[high];
                if(sum==target){
                    vector<int> temp ={nums[i],nums[low],nums[high]};
                    // sort(temp.begin(),temp.end()); // No need to sort as nums is already sorted
                    // This ensures that the triplet is in non-decreasing order
                    ans.insert(temp);
                    high--;
                    low++;
                    while(low<high && nums[low]==nums[low-1]) low++;
                    while(low<high && nums[high]== nums[high+1]) high --;
                }
                else if(sum>target){
                    high--;
                }
                else{
                    low++;
                }
            }
        }
        vector<vector<int>>final(ans.begin(),ans.end());
        return final;
    }
};
// Time Complexity: O(n^2), where n is the number of elements in the input vector.
// Space Complexity: O(n), where n is the number of unique triplets found.
class Solution1 {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        set<vector<int>>ans;
        for(int i=0;i< nums.size();i++){
            set<int> searchmap;
            for(int j=i+1;j<nums.size();j++){
                int remain = -(nums[i]+nums[j]);
                if(searchmap.find(remain)!= searchmap.end()){
                    vector<int>temp = {nums[i],nums[j],remain};
                    sort(temp.begin(),temp.end());
                    ans.insert(temp);
                }
                searchmap.insert(nums[j]);
            }
        }
        vector<vector<int>> final(ans.begin(), ans.end());
        return final;
    }
};
// Time Complexity: O(n^2), where n is the number of elements in the input vector.
// Space Complexity: O(n), where n is the number of unique triplets found.
// it give tle on test case 312 . so we can use two pointer approach