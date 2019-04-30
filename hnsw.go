package hnsw

import (
	"encoding/binary"
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
	friends    map[int][]uint32 // int:属性的ID
	attributes []string
}

type AttributeLink struct {
	IDCount    int
	attrString map[string]int // map[string]int:该属性的ID
	//attrMap     map[int][]uint32 // int:属性的ID

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

// Load opens a index file previously written by Save(). Returnes a new index and the timestamp the file was written
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
//	h.DistFunc = f32.L2Squared8AVX
//	h.bitset = bitsetpool.New()
//
//	l := readInt32(z)
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
//		h.nodes[i].level = readInt32(z)
//
//		l = readInt32(z)
//		h.nodes[i].friends = make([][]uint32, l)
//
//		for j := range h.nodes[i].friends {
//			l := readInt32(z)
//			h.nodes[i].friends[j] = make([]uint32, l)
//			err = binary.Read(z, binary.LittleEndian, h.nodes[i].friends[j])
//			if err != nil {
//				panic(err)
//			}
//		}
//
//	}
//
//	z.Close()
//	f.Close()
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
//
//	l := len(h.nodes)
//	writeInt32(l, z)
//
//	if err != nil {
//		return err
//	}
//	for _, n := range h.nodes {
//		l := len(n.p)
//		writeInt32(l, z)
//		err = binary.Write(z, binary.LittleEndian, []float32(n.p))
//		if err != nil {
//			panic(err)
//		}
//		writeInt32(n.level, z)
//
//		l = len(n.friends)
//		writeInt32(l, z)
//		for _, f := range n.friends {
//			l := len(f)
//			writeInt32(l, z)
//			err = binary.Write(z, binary.LittleEndian, f)
//			if err != nil {
//				panic(err)
//			}
//		}
//	}
//
//	z.Close()
//	f.Close()
//
//	return nil
//}

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

func (h *Hnsw) getFriends(n uint32, level int) []uint32 {
	if len(h.nodes[n].friends) < level+1 {
		return make([]uint32, 0)
	}
	return h.nodes[n].friends[level]
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

	if node.friends[attrID] == nil {
		node.friends[attrID] = make([]uint32, 0)
	}

	node.friends[attrID] = append(node.friends[attrID], second)

	l := len(node.friends[attrID])

	if l > maxL {

		// to many links, deal with it

		switch h.DelaunayType {
		case 0:
			resultSet := &distqueue.DistQueueClosestLast{Size: len(node.friends[attrID])}

			for _, n := range node.friends[attrID] {
				resultSet.Push(n, h.DistFunc(node.p, h.nodes[n].p))
			}
			for resultSet.Len() > maxL {
				resultSet.Pop()
			}
			// FRIENDS ARE STORED IN DISTANCE ORDER, closest at index 0
			node.friends[attrID] = node.friends[attrID][0:maxL]
			for i := maxL - 1; i >= 0; i-- {
				item := resultSet.Pop()
				node.friends[attrID][i] = item.ID
			}

		case 1:

			resultSet := &distqueue.DistQueueClosestFirst{Size: len(node.friends[attrID])}

			for _, n := range node.friends[attrID] {
				resultSet.Push(n, h.DistFunc(node.p, h.nodes[n].p))
			}
			h.getNeighborsByHeuristicClosestFirst(resultSet, maxL)

			// FRIENDS ARE STORED IN DISTANCE ORDER, closest at index 0
			node.friends[attrID] = node.friends[attrID][0:maxL]
			for i := 0; i < maxL; i++ {
				item := resultSet.Pop()
				node.friends[attrID][i] = item.ID
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
	h.nodes = []node{node{p: first, friends: make(map[int][]uint32)}}

	h.attributeLink = AttributeLink{
		IDCount:    0,
		attrString: make(map[string]int),
		//attrMap:     make(map[int][]uint32)
	}

	return &h
}

//func (h *Hnsw) Stats() string {
//	s := "HNSW Index\n"
//	s = s + fmt.Sprintf("M: %v, efConstruction: %v\n", h.M, h.efConstruction)
//	s = s + fmt.Sprintf("DelaunayType: %v\n", h.DelaunayType)
//	s = s + fmt.Sprintf("Number of nodes: %v\n", len(h.nodes))
//	s = s + fmt.Sprintf("Max layer: %v\n", h.maxLayer)
//	memoryUseData := 0
//	memoryUseIndex := 0
//	levCount := make([]int, h.maxLayer+1)
//	conns := make([]int, h.maxLayer+1)
//	connsC := make([]int, h.maxLayer+1)
//	for i := range h.nodes {
//		levCount[h.nodes[i].level]++
//		for j := 0; j <= h.nodes[i].level; j++ {
//			if len(h.nodes[i].friends) > j {
//				l := len(h.nodes[i].friends[j])
//				conns[j] += l
//				connsC[j]++
//			}
//		}
//		memoryUseData += h.nodes[i].p.Size()
//		memoryUseIndex += h.nodes[i].level*h.M*4 + h.M0*4
//	}
//	for i := range levCount {
//		avg := conns[i] / max(1, connsC[i])
//		s = s + fmt.Sprintf("Level %v: %v nodes, average number of connections %v\n", i, levCount[i], avg)
//	}
//	s = s + fmt.Sprintf("Memory use for data: %v (%v bytes / point)\n", memoryUseData, memoryUseData/len(h.nodes))
//	s = s + fmt.Sprintf("Memory use for index: %v (avg %v bytes / point)\n", memoryUseIndex, memoryUseIndex/len(h.nodes))
//	return s
//}

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

	ep := &distqueue.Item{ID: h.enterpoint, D: h.DistFunc(h.nodes[h.enterpoint].p, q)}

	// assume Grow has been called in advance
	newID := id
	newNode := node{p: q, friends: make(map[int][]uint32), attributes: attributes}

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
		if _, ok := h.attributeLink.attrString[attrString]; ok {
			//if attrString != "nil;nil;nil" {
			//	fmt.Print("存在ID:")
			//	fmt.Println(attrString)
			//}
			attrID = h.attributeLink.attrString[attrString]
			//h.attributeLink.attrMap[attrID] = append(h.attributeLink.attrMap[attrID], newID)

			resultSet := &distqueue.DistQueueClosestLast{}
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

			newNode.friends[attrID] = make([]uint32, resultSet.Len())
			for i := resultSet.Len() - 1; i >= 0; i-- {
				item := resultSet.Pop()
				// store in order, closest at index 0
				newNode.friends[attrID][i] = item.ID
			}
		} else {
			//fmt.Print("插入新ID:")
			//fmt.Println(attrString)
			attrID = h.attributeLink.IDCount
			h.attributeLink.attrString[attrString] = attrID
			h.nodes[0].friends[attrID] = append(h.nodes[0].friends[attrID], newID)
			//h.attributeLink.attrMap[attrID] = append(h.attributeLink.attrMap[attrID], newID)
			h.attributeLink.IDCount += 1
		}

		h.Lock()
		// Add it and increase slice length if neccessary
		if len(h.nodes) < int(newID)+1 {
			h.nodes = h.nodes[0 : newID+1]
		}
		h.nodes[newID] = newNode
		h.Unlock()

		// now add connections to newNode from newNodes neighbours (makes it visible in the graph)
		//for level := min(curlevel, currentMaxLayer); level >= 0; level-- {
		//	for _, n := range newNode.friends[level] {
		//		h.Link(n, newID, level)
		//	}
		//}

		for _, n := range newNode.friends[attrID] {
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

		friends := h.nodes[c.ID].friends[attrID]

		//if attrID != 0{
		//	fmt.Println(friends)
		//	for _, item := range friends {
		//		fmt.Println(h.nodes[item].attributes)
		//	}
		//}

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
func (h *Hnsw) SearchBrute(q Point, K int) *distqueue.DistQueueClosestLast {
	resultSet := &distqueue.DistQueueClosestLast{Size: K}
	for i := 1; i < len(h.nodes); i++ {
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
	groundTruth := h.SearchBrute(q, K)
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

	h.RLock()
	ep := &distqueue.Item{ID: h.enterpoint, D: h.DistFunc(h.nodes[h.enterpoint].p, q)}
	h.RUnlock()

	resultSet := &distqueue.DistQueueClosestLast{Size: ef + 1}

	attrString := ""
	for i, attr := range attributes {
		attrString += attr
		if i < len(attributes)-1 {
			attrString += ";"
		}
	}

	attrID := h.attributeLink.attrString[attrString]
	//fmt.Println("&&&&&&&")
	//fmt.Println(attrString)
	//fmt.Println(attrID)
	h.searchAtLayer(q, resultSet, ef, ep, attrID)

	for resultSet.Len() > K {
		resultSet.Pop()
	}
	return resultSet
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
