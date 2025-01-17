import heapq


# Authors:
#   Edward L. Platt <ed@elplatt.com>
# Edited by
#   Hao Hu
class HashPQueue(object):
    def __init__(self, data=[]):
        """Priority queue class with updatable priorities.
        """
        self.h = list(data)
        self.d = dict()
        self._heapify()
        self.rmax = 100000

    def __len__(self):
        return len(self.h)

    def _heapify(self):
        """Restore heap invariant and recalculate map."""
        heapq.heapify(self.h)
        self.d = dict([(elt[1], pos) for pos, elt in enumerate(self.h)])
        if len(self.h) != len(self.d):
            raise AssertionError("Heap contains duplicate elements")

    def push(self, priority, item):
        """Add an element to the queue."""
        # If element is already in queue, do nothing
        if item in self.d:
            self.update(priority, item)
        else:
            # Add element to heap and dict
            pos = len(self.h)
            self.h.append((priority, item))
            self.d[item] = pos
            # Restore invariant by sifting down
            self._siftdown(pos)

    def pop(self):
        """Remove and return the largest element in the queue."""
        # Remove smallest element
        elt = self.h[0]
        del self.d[elt[1]]
        # If elt is last item, remove and return
        if len(self.h) == 1:
            self.h.pop()
            return elt
        # Replace root with last element
        last = self.h.pop()
        self.h[0] = last
        self.d[last[1]] = 0
        # Restore invariant by sifting up, then down
        pos = self._siftup(0)
        self._siftdown(pos)
        # Return smallest element
        return elt

    def update(self, new_priority, item):
        """Replace an element in the queue with a new one."""
        # Replace
        pos = self.d[item]
        old_priority = self.h[pos][0]
        if max(new_priority, old_priority) >= self.rmax:
            new_priority = max(new_priority, old_priority)
        self.h[pos] = (new_priority, item)
        # Restore invariant by sifting up, then down
        pos = self._siftup(pos)
        self._siftdown(pos)

    def remove(self, item):
        """Remove an element from the queue."""
        # Find and remove element
        try:
            pos = self.d[item]
            del self.d[item]
        except KeyError:
            # Not in queue
            return
        # If elt is last item, remove and return
        if pos == len(self.h) - 1:
            self.h.pop()
            return
        # Replace elt with last element
        last = self.h.pop()
        self.h[pos] = last
        self.d[last[1]] = pos
        # Restore invariant by sifting up, then down
        pos = self._siftup(pos)
        self._siftdown(pos)

    def _siftup(self, pos):
        """Move element at pos down to a leaf by repeatedly moving the smaller
        child up."""
        h, d = self.h, self.d
        elt = h[pos]
        # priority, item = elt
        # Continue until element is in a leaf
        end_pos = len(h)
        left_pos = (pos << 1) + 1
        while left_pos < end_pos:
            # Left child is guaranteed to exist by loop predicate
            left = h[left_pos]
            try:
                right_pos = left_pos + 1
                right = h[right_pos]
                # Out-of-place, swap with left unless right is smaller
                if right[0] > left[0]:
                    h[pos], h[right_pos] = right, elt
                    pos, right_pos = right_pos, pos
                    d[elt[1]], d[right[1]] = pos, right_pos
                else:
                    h[pos], h[left_pos] = left, elt
                    pos, left_pos = left_pos, pos
                    d[elt[1]], d[left[1]] = pos, left_pos
            except IndexError:
                # Left leaf is the end of the heap, swap
                h[pos], h[left_pos] = left, elt
                pos, left_pos = left_pos, pos
                d[elt[1]], d[left[1]] = pos, left_pos
            # Update left_pos
            left_pos = (pos << 1) + 1
        return pos

    def _siftdown(self, pos):
        """Restore invariant by repeatedly replacing out-of-place element with
        its parent."""
        h, d = self.h, self.d
        elt = h[pos]
        # Continue until element is at root
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            parent = h[parent_pos]
            if parent[0] < elt[0]:
                # Swap out-of-place element with parent
                h[parent_pos], h[pos] = elt, parent
                parent_pos, pos = pos, parent_pos
                d[elt[1]] = pos
                d[parent[1]] = parent_pos
            else:
                # Invariant is satisfied
                break
        return pos
