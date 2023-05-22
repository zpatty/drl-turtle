from bisect import bisect

class ShortList:
    def __init__(self, size, min=True):
        self.size = size
        self._sorted_items = []
        self._sorted_values = []

    def front(self):
        return self._sorted_items[0], self._sorted_values[0]
    
    def back(self):
        return self._sorted_items[-1], self._sorted_values[-1]
    
    def insert(self, item, value):
        i = bisect(self._sorted_values, value)

        if i==self.size:
            return False

        self._sorted_items.insert(i, item)
        self._sorted_values.insert(i, value)
        
        if (len(self._sorted_items) > self.size):
            self._sorted_items.pop()
            self._sorted_values.pop()
        return True


"""

Quick tests

"""
if __name__ == "__main__":
    short_list = ShortList(3)
    def insert_print(item, value):
        print(f'inserting {item} with value {value}')
        short_list.insert(item, value)
        print(f'Front is now: {short_list.front()}\t and back is now: {short_list.back()} ')
    
    insert_print("a", 5)
    insert_print("hello", 3)
    insert_print("very high", 1000)
    insert_print("very_low", -1000)
    insert_print("zero", 0)
    insert_print("kinda high", 100)
    insert_print("kinda_low", -100)