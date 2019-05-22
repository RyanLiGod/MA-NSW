package hnsw

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	//"fmt"
	"io"

	"./bitsetpool"
	"./distqueue"
	"./f32"
	"sync"
)

type Point []float32

func (a Point) Size() int {
	return len(a) * 4
}

type node struct {
	sync.RWMutex
	locked     bool
	p          Point
	friends    sync.Map // map[int][]uint32 int:属性的ID
	attributes []string
}

type AttributeLink struct {
	IDCount    int
	attrString sync.Map // map[string]int int:该属性的ID
}

type Hnsw struct {
	sync.RWMutex
	M              int
	efConstruction int
	linkMode       int
	DelaunayType   int
	DistFunc       func([]float32, []float32) float32
	nodes          []node
	bitset         *bitsetpool.BitsetPool
	enterpoint     uint32
	attributeLink  AttributeLink
}

// Load opens a index file previously written by Save(). Returns a new index and the timestamp the file was written
//func Load(filename string) (*Hnsw, int64, error) {
//	f, err := os.Open(filename)
//	if err != nil {
//		return nil, 0, err
//	}
//	z, err := gzip.NewReader(f)
//	if err != nil {
//		return nil, 0, err
//	}
//
//	timestamp := readInt64(z)
//
//	h := new(Hnsw)
//	h.M = readInt32(z)
//	h.efConstruction = readInt32(z)
//	h.linkMode = readInt32(z)
//	h.DelaunayType = readInt32(z)
//	h.enterpoint = uint32(readInt32(z))
//
//	h.attributeLink.IDCount = readInt32(z)
//	l := int(readInt32(z))
//	h.attributeLink.attrString = readAttrString(z, l)
//
//	h.DistFunc = f32.L2Squared8AVX
//	h.bitset = bitsetpool.New()
//
//	l = readInt32(z)
//	h.nodes = make([]node, l)
//
//	for i := range h.nodes {
//
//		l := readInt32(z)
//		h.nodes[i].p = make([]float32, l)
//
//		err = binary.Read(z, binary.LittleEndian, h.nodes[i].p)
//		if err != nil {
//			panic(err)
//		}
//
//		// Read friends
//		l = readInt32(z)
//		bt := make([]byte, int(l))
//		err := binary.Read(z, binary.LittleEndian, &bt)
//		if err != nil {
//			panic(err)
//		}
//		var friends map[int][]uint32
//		err = json.Unmarshal(bt, &friends)
//		if err != nil {
//			panic(err)
//		}
//		h.nodes[i].friends = friends
//
//		// Read attributes (can be deleted in product mode)
//		l = readInt32(z)
//		bt = make([]byte, int(l))
//		err = binary.Read(z, binary.LittleEndian, &bt)
//		if err != nil {
//			panic(err)
//		}
//		var attributes []string
//		err = json.Unmarshal(bt, &attributes)
//		if err != nil {
//			panic(err)
//		}
//		h.nodes[i].attributes = attributes
//	}
//
//	_ = z.Close()
//	_ = f.Close()
//
//	return h, timestamp, nil
//}

// Save writes to current index to a gzipped binary data file
//func (h *Hnsw) Save(filename string) error {
//	f, err := os.Create(filename)
//	if err != nil {
//		return err
//	}
//	z := gzip.NewWriter(f)
//
//	timestamp := time.Now().Unix()
//
//	writeInt64(timestamp, z)
//
//	writeInt32(h.M, z)
//	writeInt32(h.efConstruction, z)
//	writeInt32(h.linkMode, z)
//	writeInt32(h.DelaunayType, z)
//	writeInt32(int(h.enterpoint), z)
//	writeInt32(h.attributeLink.IDCount, z)
//	writeAttrString(h.attributeLink.attrString, z)
//
//	l := len(h.nodes)
//	writeInt32(l, z)
//
//	for _, n := range h.nodes {
//		l := len(n.p)
//		writeInt32(l, z)
//		err = binary.Write(z, binary.LittleEndian, []float32(n.p))
//		if err != nil {
//			panic(err)
//		}
//
//		// Write friends
//		res, err := json.Marshal(n.friends)
//		if err != nil {
//			panic(err)
//		}
//		err = binary.Write(z, binary.LittleEndian, int32(len(res)))
//		if err != nil {
//			panic(err)
//		}
//		err = binary.Write(z, binary.LittleEndian, &res)
//		if err != nil {
//			panic(err)
//		}
//
//		// Write attributes (can be deleted in product mode)
//		res, err = json.Marshal(n.attributes)
//		if err != nil {
//			panic(err)
//		}
//		err = binary.Write(z, binary.LittleEndian, int32(len(res)))
//		if err != nil {
//			panic(err)
//		}
//		err = binary.Write(z, binary.LittleEndian, &res)
//		if err != nil {
//			panic(err)
//		}
//	}
//
//	_ = z.Close()
//	_ = f.Close()
//
//	return nil
//}

func writeAttrString(v map[string]int, w io.Writer) {
	res, err := json.Marshal(&v)
	if err != nil {
		panic(err)
	}
	err = binary.Write(w, binary.LittleEndian, int32(len(res)))
	if err != nil {
		panic(err)
	}
	err = binary.Write(w, binary.LittleEndian, &res)
	if err != nil {
		panic(err)
	}
}

func readAttrString(r io.Reader, lenByte int) map[string]int {
	i := make([]byte, int(lenByte))
	err := binary.Read(r, binary.LittleEndian, &i)
	if err != nil {
		panic(err)
	}
	var attrLink map[string]int
	err = json.Unmarshal(i, &attrLink)
	if err != nil {
		panic(err)
	}
	return attrLink
}

func writeInt64(v int64, w io.Writer) {
	err := binary.Write(w, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
}

func writeInt32(v int, w io.Writer) {
	i := int32(v)
	err := binary.Write(w, binary.LittleEndian, &i)
	if err != nil {
		panic(err)
	}
}

func readInt32(r io.Reader) int {
	var i int32
	err := binary.Read(r, binary.LittleEndian, &i)
	if err != nil {
		panic(err)
	}
	return int(i)
}

func writeFloat64(v float64, w io.Writer) {
	err := binary.Write(w, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
}

func readInt64(r io.Reader) (v int64) {
	err := binary.Read(r, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
	return
}

func readFloat64(r io.Reader) (v float64) {
	err := binary.Read(r, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
	return
}

func (h *Hnsw) getFriends(n uint32, attrID int) []uint32 {
	_friends, _ := h.nodes[n].friends.Load(attrID)
	return _friends.([]uint32)
}

func (h *Hnsw) Link(first, second uint32, attrID int) {
	// fmt.Printf("first: %d\n", first)
	// fmt.Printf("second: %d\n", second)
	// fmt.Printf("level: %d\n", level)
	// fmt.Printf("------------------------\n")
	maxL := h.M

	h.RLock()
	node := &h.nodes[first]
	h.RUnlock()

	node.Lock()

	//// check if we have allocated friends slices up to this level?
	//if len(node.friends) < h.attributeLink.IDCount+1 {
	//	for j := len(node.friends); j <= h.attributeLink.IDCount; j++ {
	//		// allocate new list with 0 elements but capacity maxL
	//		node.friends = append(node.friends, make([]uint32, 0, maxL))
	//	}
	//	// now grow it by one and add the first connection for this layer
	//	node.friends[level] = node.friends[level][0:1]
	//	node.friends[level][0] = second
	//
	//} else {
	//	// we did have some already... this will allocate more space if it overflows maxL
	//	node.friends[attrID] = append(node.friends[attrID], second)
	//}

	//if node.friends == nil {
	//	node.friends = make(map[int][]uint32)
	//}

	_temp, _ := node.friends.Load(attrID)
	_friends := _temp.([]uint32)
	if _friends == nil {
		node.friends.Store(attrID, make([]uint32, 0))
	}

	_friends = append(_friends, second)
	node.friends.Store(attrID, _friends)

	l := len(_friends)

	if l > maxL {

		// to many links, deal with it

		switch h.DelaunayType {
		case 0:
			resultSet := &distqueue.DistQueueClosestLast{Size: l}

			for _, n := range _friends {
				resultSet.Push(n, h.DistFunc(node.p, h.nodes[n].p))
			}
			for resultSet.Len() > maxL {
				resultSet.Pop()
			}
			// FRIENDS ARE STORED IN DISTANCE ORDER, closest at index 0
			node.friends.Store(attrID, _friends[0:maxL])
			for i := maxL - 1; i >= 0; i-- {
				item := resultSet.Pop()
				_temp, _ := node.friends.Load(attrID)
				_friends := _temp.([]uint32)
				_friends[i] = item.ID
				node.friends.Store(attrID, _friends)
			}

		case 1:

			resultSet := &distqueue.DistQueueClosestFirst{Size: l}

			for _, n := range _friends {
				resultSet.Push(n, h.DistFunc(node.p, h.nodes[n].p))
			}
			h.getNeighborsByHeuristicClosestFirst(resultSet, maxL)

			// FRIENDS ARE STORED IN DISTANCE ORDER, closest at index 0
			node.friends.Store(attrID, _friends[0:maxL])
			for i := 0; i < maxL; i++ {
				item := resultSet.Pop()
				_temp, _ := node.friends.Load(attrID)
				_friends := _temp.([]uint32)
				_friends[i] = item.ID
				node.friends.Store(attrID, _friends)
			}
		}
	}
	node.Unlock()
}

func (h *Hnsw) getNeighborsByHeuristicClosestLast(resultSet1 *distqueue.DistQueueClosestLast, M int) {
	if resultSet1.Len() <= M {
		return
	}
	resultSet := &distqueue.DistQueueClosestFirst{Size: resultSet1.Len()}
	tempList := &distqueue.DistQueueClosestFirst{Size: resultSet1.Len()}
	result := make([]*distqueue.Item, 0, M)
	for resultSet1.Len() > 0 {
		resultSet.PushItem(resultSet1.Pop())
	}
	for resultSet.Len() > 0 {
		if len(result) >= M {
			break
		}
		e := resultSet.Pop()
		good := true
		for _, r := range result {
			if h.DistFunc(h.nodes[r.ID].p, h.nodes[e.ID].p) < e.D {
				good = false
				break
			}
		}
		if good {
			result = append(result, e)
		} else {
			tempList.PushItem(e)
		}
	}
	for len(result) < M && tempList.Len() > 0 {
		result = append(result, tempList.Pop())
	}
	for _, item := range result {
		resultSet1.PushItem(item)
	}
}

func (h *Hnsw) getNeighborsByHeuristicClosestFirst(resultSet *distqueue.DistQueueClosestFirst, M int) {
	if resultSet.Len() <= M {
		return
	}
	tempList := &distqueue.DistQueueClosestFirst{Size: resultSet.Len()}
	result := make([]*distqueue.Item, 0, M)
	for resultSet.Len() > 0 {
		if len(result) >= M {
			break
		}
		e := resultSet.Pop()
		good := true
		for _, r := range result {
			if h.DistFunc(h.nodes[r.ID].p, h.nodes[e.ID].p) < e.D {
				good = false
				break
			}
		}
		if good {
			result = append(result, e)
		} else {
			tempList.PushItem(e)
		}
	}
	for len(result) < M && tempList.Len() > 0 {
		result = append(result, tempList.Pop())
	}
	resultSet.Reset()

	for _, item := range result {
		resultSet.PushItem(item)
	}
}

func (h *Hnsw) GetAttributeLink() AttributeLink {
	return h.attributeLink
}

func (h *Hnsw) GetNodes() []node {
	return h.nodes
}

func New(M int, efConstruction int, first Point) *Hnsw {

	h := Hnsw{}
	h.M = M
	// default values used in c++ implementation
	h.efConstruction = efConstruction
	h.DelaunayType = 1

	h.bitset = bitsetpool.New()

	h.DistFunc = f32.L2Squared8AVX

	// add first point, it will be our enterpoint (index 0)
	h.nodes = []node{{p: first, friends: sync.Map{}}}

	h.attributeLink = AttributeLink{
		IDCount:    0,
		attrString: sync.Map{},
	}

	return &h
}

func (h *Hnsw) Stats() string {
	s := "MA-NSW Index\n"
	s = s + fmt.Sprintf("M: %v, efConstruction: %v\n", h.M, h.efConstruction)
	s = s + fmt.Sprintf("DelaunayType: %v\n", h.DelaunayType)
	s = s + fmt.Sprintf("Number of nodes: %v\n", len(h.nodes)-1)
	attrNum := h.attributeLink.IDCount
	memoryUseData := 0
	memoryUseIndex := 0
	attrCount := make(map[string]int, attrNum)
	//conns := make(map[int]int, attrNum)
	//connsC := make(map[int]int, attrNum)
	for i := range h.nodes {
		attrString := ""
		for m, attr := range h.nodes[i].attributes {
			attrString += attr
			if m < len(h.nodes[i].attributes)-1 {
				attrString += ";"
			}
		}
		//attrID := h.attributeLink.attrString[attrString]
		if attrString != "" {
			attrCount[attrString]++
		}
		//for k := range h.nodes[i].friends {
		//	l := len(h.nodes[i].friends[k])
		//	conns[k] += l
		//	connsC[k]++
		//}
		memoryUseData += h.nodes[i].p.Size()
		_len := 0
		h.nodes[i].friends.Range(func(k,v interface{})bool{
			_len++
			return true
		})
		memoryUseIndex += _len*h.M*4
	}
	for i := range attrCount {
		//avg := conns[i] / max(1, connsC[i])
		s = s + fmt.Sprintf("Attributes %v: %v nodes\n", i, attrCount[i])
	}
	s = s + fmt.Sprintf("Memory use for data: %v (%v bytes / point)\n", memoryUseData, memoryUseData/len(h.nodes))
	s = s + fmt.Sprintf("Memory use for index: %v (avg %v bytes / point)\n", memoryUseIndex, memoryUseIndex/len(h.nodes))
	return s
}

func (h *Hnsw) Grow(size int) {
	// fmt.Println(h.nodes)
	if size+1 <= len(h.nodes) {
		return
	}
	newNodes := make([]node, len(h.nodes), size+1)
	copy(newNodes, h.nodes)
	h.nodes = newNodes
}

func (h *Hnsw) GetNodeAttr(id uint32) []string {
	return h.nodes[id].attributes
}

func (h *Hnsw) Add(q Point, id uint32, attributes []string) {
	if id == 0 {
		panic("Id 0 is reserved, use ID:s starting from 1 when building index")
	}

	h.enterpoint = 0

	// assume Grow has been called in advance
	newID := id
	newNode := node{p: q, friends: sync.Map{}, attributes: attributes}

	n := uint(len(attributes))
	var maxCount uint = 1 << n
	var i uint
	var j uint
	for i = 0; i < maxCount; i++ {
		attrString := ""
		for j = 0; j < n; j++ {
			if (i & (1 << j)) != 0 {
				attrString += attributes[j]
			} else {
				attrString += "nil"
			}
			if j < n-1 {
				attrString += ";"
			}
		}

		attrID := 0
		if _, ok := h.attributeLink.attrString.Load(attrString); ok {
			//if attrString != "nil;nil;nil" {
			//	fmt.Print("存在ID:")
			//	fmt.Println(attrString)
			//}
			_attrID, _ := h.attributeLink.attrString.Load(attrString)
			attrID = _attrID.(int)
			//h.attributeLink.attrMap[attrID] = append(h.attributeLink.attrMap[attrID], newID)

			resultSet := &distqueue.DistQueueClosestLast{}

			_epID, _ := h.nodes[h.enterpoint].friends.Load(attrID)
			epID := _epID.([]uint32)[0]
			h.RLock()
			ep := &distqueue.Item{ID: epID, D: h.DistFunc(h.nodes[epID].p, q)}
			h.RUnlock()

			h.searchAtLayer(q, resultSet, h.efConstruction, ep, attrID)
			switch h.DelaunayType {
			case 0:
				// shrink resultSet to M closest elements (the simple heuristic)
				for resultSet.Len() > h.M {
					resultSet.Pop()
				}
			case 1:
				h.getNeighborsByHeuristicClosestLast(resultSet, h.M)
			}

			newNode.friends.Store(attrID, make([]uint32, resultSet.Len()))
			for i := resultSet.Len() - 1; i >= 0; i-- {
				item := resultSet.Pop()
				// store in order, closest at index 0
				_temp, _ := newNode.friends.Load(attrID)
				f := _temp.([]uint32)
				f[i] = item.ID
				newNode.friends.Store(attrID, f)
			}
		} else {
			//fmt.Print("插入新ID:")
			//fmt.Println(attrString)
			attrID = h.attributeLink.IDCount
			h.attributeLink.attrString.Store(attrString, attrID)
			h.nodes[0].friends.Store(attrID, []uint32{newID})
			newNode.friends.Store(attrID, make([]uint32, 0))
			h.attributeLink.IDCount += 1
		}

		h.Lock()
		// Add it and increase slice length if neccessary
		if len(h.nodes) < int(newID)+1 {
			h.nodes = h.nodes[0 : newID+1]
		}
		h.nodes[newID] = newNode
		h.Unlock()

		friends, _ := newNode.friends.Load(attrID)
		for _, n := range friends.([]uint32) {
			h.Link(n, newID, attrID)
		}
	}
}

func (h *Hnsw) searchAtLayer(q Point, resultSet *distqueue.DistQueueClosestLast, efConstruction int, ep *distqueue.Item, attrID int) {

	var pool, visited = h.bitset.Get()
	//visited := make(map[uint32]bool)

	candidates := &distqueue.DistQueueClosestFirst{Size: efConstruction * 3}

	visited.Set(uint(ep.ID))
	//visited[ep.ID] = true
	candidates.Push(ep.ID, ep.D)

	resultSet.Push(ep.ID, ep.D)

	for candidates.Len() > 0 {
		_, lowerBound := resultSet.Top() // worst distance so far
		c := candidates.Pop()

		if c.D > lowerBound {
			// since candidates is sorted, it wont get any better...
			break
		}

		_friends, _ := h.nodes[c.ID].friends.Load(attrID)
		friends := _friends.([]uint32)

		for _, n := range friends {
			if !visited.Test(uint(n)) {
				visited.Set(uint(n))
				d := h.DistFunc(q, h.nodes[n].p)
				_, topD := resultSet.Top()
				if resultSet.Len() < efConstruction {
					item := resultSet.Push(n, d)
					candidates.PushItem(item)
				} else if topD > d {
					// keep length of resultSet to max efConstruction
					item := resultSet.PopAndPush(n, d)
					candidates.PushItem(item)
				}
			}
		}
	}
	h.bitset.Free(pool)
}

// SearchBrute returns the true K nearest neigbours to search point q
func (h *Hnsw) SearchBrute(q Point, K int, attributes []string) *distqueue.DistQueueClosestLast {
	resultSet := &distqueue.DistQueueClosestLast{Size: K}
	for i := 1; i < len(h.nodes); i++ {
		if !stringSliceEqual(h.nodes[i].attributes, attributes) {
			continue
		}
		d := h.DistFunc(h.nodes[i].p, q)
		if resultSet.Len() < K {
			resultSet.Push(uint32(i), d)
			continue
		}
		_, topD := resultSet.Head()
		if d < topD {
			resultSet.PopAndPush(uint32(i), d)
			continue
		}
	}
	return resultSet
}

// Benchmark test precision by comparing the results of SearchBrute and Search
func (h *Hnsw) Benchmark(q Point, ef int, K int, attributes []string) float64 {
	result := h.Search(q, ef, K, attributes)
	groundTruth := h.SearchBrute(q, K, attributes)
	truth := make([]uint32, 0)
	for groundTruth.Len() > 0 {
		truth = append(truth, groundTruth.Pop().ID)
	}
	p := 0
	for result.Len() > 0 {
		i := result.Pop()
		for j := 0; j < K; j++ {
			if truth[j] == i.ID {
				p++
			}
		}
	}
	return float64(p) / float64(K)
}

func (h *Hnsw) Search(q Point, ef int, K int, attributes []string) *distqueue.DistQueueClosestLast {

	resultSet := &distqueue.DistQueueClosestLast{Size: ef + 1}

	attrString := ""
	for i, attr := range attributes {
		attrString += attr
		if i < len(attributes)-1 {
			attrString += ";"
		}
	}

	if _, ok := h.attributeLink.attrString.Load(attrString); ok {
		_attrID, _ := h.attributeLink.attrString.Load(attrString)
		attrID := _attrID.(int)
		_epID, _ := h.nodes[h.enterpoint].friends.Load(attrID)
		epID := _epID.([]uint32)[0]
		h.RLock()
		ep := &distqueue.Item{ID: epID, D: h.DistFunc(h.nodes[epID].p, q)}
		h.RUnlock()
		h.searchAtLayer(q, resultSet, ef, ep, attrID)

		for resultSet.Len() > K {
			resultSet.Pop()
		}
		return resultSet
	} else {
		return &distqueue.DistQueueClosestLast{Size: 0}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}

	if (a == nil) != (b == nil) {
		return false
	}

	b = b[:len(a)]
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}

	return true
}