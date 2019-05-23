package main

import (
	hnsw ".."
	"fmt"
	"github.com/grd/stat"
	"math/rand"
	"time"
)

// NUM 元素数量
var NUM = 3000

// DIMENSION 元素维度
var DIMENSION = 128

// TESTNUM 测试数量
var TESTNUM = 10

func main() {

	const (
		M              = 16
		efConstruction = 400
		efSearch       = 100
		K              = 10
	)

	var zero hnsw.Point = make([]float32, DIMENSION)

	//h := hnsw.New(M, efConstruction, zero, "l2")
	h := hnsw.New(M, efConstruction, zero, "cosine")
	h.Grow(NUM)

	provinces := []string{"浙江省", "江西省", "安徽省"}
	types := []string{"高校", "企业", "其他"}
	titles := []string{"教授", "讲师"}

	for i := 1; i <= NUM; i++ {
		//fmt.Println("--------------------")
		//fmt.Println(i)
		randomAttr := []string{provinces[rand.Intn(3)], types[rand.Intn(3)], titles[rand.Intn(2)]}
		//fmt.Println(randomAttr)
		h.Add(randomPoint(), uint32(i), randomAttr)
		// h.Add(randomPoint(), uint32(i))
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
		//fmt.Println(h.GetNodes()[0])
	}
	//fmt.Println(h.GetAttributeLink())
	//fmt.Println(h.GetNodes()[0])
	//fmt.Println(h)

	//_ = h.Save("test.ind")

	//h, timestamp, _ := hnsw.Load("test.ind")
	//fmt.Printf("Index loaded, time saved was %v\n", time.Unix(timestamp, 0))


	fmt.Printf("Now searching with HNSW...\n")
	timeRecord := make([]float64, TESTNUM)
	hits := 0
	// start := time.Now()
	for i := 0; i < TESTNUM; i++ {
		searchAttr := []string{provinces[rand.Intn(3)], types[rand.Intn(3)], titles[rand.Intn(2)]}
		fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
		queries := make([]hnsw.Point, TESTNUM)
		truth := make([][]uint32, TESTNUM)
		for i := range queries {
			queries[i] = randomPoint()
			result := h.SearchBrute(queries[i], K, searchAttr)
			truth[i] = make([]uint32, K)
			for j := K - 1; j >= 0; j-- {
				item := result.Pop()
				truth[i][j] = item.ID
			}
		}
		startSearch := time.Now()
		result := h.Search(queries[i], efSearch, K, searchAttr)
		//result := h.Search(queries[i], efSearch, K, []string{"nil", "nil", "nil"})
		fmt.Print("Searching with attributes:")
		fmt.Println(searchAttr)
		stopSearch := time.Since(startSearch)
		timeRecord[i] = stopSearch.Seconds() * 1000
		if result.Size != 0 {
			for j := 0; j < K; j++ {
				item := result.Pop()
				fmt.Printf("%v  ", item)
				if item != nil {
					fmt.Println(h.GetNodeAttr(item.ID))
					for k := 0; k < K; k++ {
						if item.ID == truth[i][k] {
							hits++
						}
					}
				}
			}
		} else {
			fmt.Println("Can't return any node")
		}

		fmt.Println()
	}

	//fmt.Println(h.GetNodes()[221])

	// stop := time.Since(start)

	data := stat.Float64Slice(timeRecord)
	mean := stat.Mean(data)
	variance := stat.Variance(data)

	fmt.Printf("Mean of queries time(MS): %v\n", mean)
	fmt.Printf("Variance of queries time: %v\n", variance)
	fmt.Printf("%v queries / second (single thread)\n", 1000.0/mean)
	fmt.Printf("Average 10-NN precision: %v\n", float64(hits)/(float64(TESTNUM)*float64(K)))
	fmt.Printf("\n")
	fmt.Printf(h.Stats())
}

func randomPoint() hnsw.Point {
	var v hnsw.Point = make([]float32, DIMENSION)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
