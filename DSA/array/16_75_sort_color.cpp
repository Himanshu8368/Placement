#include <vector>
using namespace std;
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int ones =0;
        int twos = 0;
        int zeros =0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==0){
                zeros++;
            }
            else if (nums[i]==1){
                ones++;
            }
            else{
                twos++;
            }
        }
        int index =0;
        for(int i=index;i<zeros;i++){
            nums[i]=0;
        }
        index += zeros;
        for(int i=index;i<index+ones;i++){
            nums[i]=1;
        }
        index += ones;
        for(int i=index;i<index+twos;i++){
            nums[i]=2;
        }
    }
};
// This code implements a solution to the "Sort Colors" problem, which is a common interview question. The problem requires sorting an array containing only 0s, 1s, and 2s in a single pass with O(n) time complexity and O(1) space complexity. The first solution uses counting to sort the colors, while the second solution uses the Dutch National Flag algorithm for in-place sorting.
// The first solution counts the occurrences of each color and then fills the array accordingly, while the second solution uses a two-pointer approach to sort the colors in a single pass.
class Solution2 {
public:
    void sortColors(vector<int>& nums) {
        int low ,mid =0;
        int high = nums.size()-1;
        while(mid<=high){
            if(nums[mid]==0){
                swap(nums[mid],nums[low]);
                low++;
                mid++;
            }
            else if(nums[mid]==2){
                swap(nums[high],nums[mid]);
                high--;
            }
            else{
                mid++;
            }
        }
    }
};

// This code implements the Dutch National Flag problem solution, which sorts an array containing 0s, 1s, and 2s in a single pass.
// The algorithm uses three pointers: low, mid, and high. It iterates through the array and swaps elements to ensure that all 0s are at the beginning, all 1s are in the middle, and all 2s are at the end.