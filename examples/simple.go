package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/grd/stat"

	hnsw ".."
)

// NUM 元素数量
var NUM = 50000

// DIMENSION 元素维度
var DIMENSION = 128

// TESTNUM 测试数量
var TESTNUM = 1000

func main() {

	const (
		M              = 100
		efConstruction = 1000
		efSearch       = 1000
		K              = 10
	)

	var zero hnsw.Point = make([]float32, DIMENSION)

	h := hnsw.New(M, efConstruction, zero)
	h.Grow(NUM)

	for i := 1; i <= NUM; i++ {
		h.BalancedAdd(randomPoint(), uint32(i))
		// h.Add(randomPoint(), uint32(i))
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}
	// h.Save("BalancedAdd_100000p_128d_64M_1000efc.ind")

	// h, timestamp := hnsw.Load("BalancedAdd_50000p_128d_100M_2000efc.ind")
	// h, timestamp := hnsw.Load("Add_50000p_128d_100M_2000efc.ind")
	// fmt.Printf("Index loaded, time saved was %v\n", time.Unix(timestamp, 0))

	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
	queries := make([]hnsw.Point, TESTNUM)
	truth := make([][]uint32, TESTNUM)
	for i := range queries {
		queries[i] = randomPoint()
		result := h.SearchBrute(queries[i], K)
		truth[i] = make([]uint32, K)
		for j := K - 1; j >= 0; j-- {
			item := result.Pop()

			truth[i][j] = item.ID
		}
	}

	fmt.Printf("Now searching with HNSW...\n")
	timeRecord := make([]float64, TESTNUM)
	hits := 0
	start := time.Now()
	for i := 0; i < TESTNUM; i++ {
		startSearch := time.Now()
		result := h.Search(queries[i], efSearch, K)
		stopSearch := time.Since(startSearch)
		timeRecord[i] = stopSearch.Seconds() * 1000
		for j := 0; j < K; j++ {
			item := result.Pop()
			for k := 0; k < K; k++ {
				if item.ID == truth[i][k] {
					hits++
				}
			}
		}
	}
	stop := time.Since(start)

	data := stat.Float64Slice(timeRecord)
	mean := stat.Mean(data)
	variance := stat.Variance(data)

	fmt.Printf("Mean of queries time: %v\n", mean)
	fmt.Printf("Variance of queries time: %v\n", variance)
	fmt.Printf("%v queries / second (single thread)\n", 1000.0/stop.Seconds())
	fmt.Printf("Average 10-NN precision: %v\n", float64(hits)/(1000.0*float64(K)))
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
