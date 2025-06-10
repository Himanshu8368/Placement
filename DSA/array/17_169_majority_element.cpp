#include <vector>
#include <map>  
using namespace std;

class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int n = nums.size();
        int element;
        int count =0;
        for(int i=0;i<nums.size();i++){
            if(count ==0){
                count++;
                element = nums[i];
            }
            else if (element == nums[i]) count ++;
            else count --;
        }
        int check=0;
        for(int i=0;i<n ;i++){
            if(element == nums[i])  check++;
        }
        if(check >(n/2)) return element;
        return -1;
    }
    // int majorityElement(vector<int>& nums) {
    //     map<int,int> mpp;
    //     int n = nums.size();
    //     for(int i=0;i<n;i++){
    //         mpp[nums[i]]++;
    //     }
    //     for(auto it : mpp){
    //         if(it.second > (n/2)){
    //             return it.first;
    //         }
    //     }
    //     return -1;
    // }
};
// Time Complexity: O(n)
// Space Complexity: O(1) for the first approach, O(n) for the second approach using map

