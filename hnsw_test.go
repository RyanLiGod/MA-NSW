package hnsw

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grd/stat"
	"github.com/stretchr/testify/assert"
)

// var prefix = "sift/sift"
var prefix = "siftsmall/siftsmall"
var dataSize = 1000000
var efSearch = []int{1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 400, 1000}
var queries []Point
var truth [][]uint32

func TestMain(m *testing.M) {
	// LOAD QUERIES AND GROUNDTRUTH
	fmt.Printf("Loading query records\n")
	queries, truth = loadQueriesFromFvec(prefix)
	os.Exit(m.Run())
}
func TestSaveLoad(t *testing.T) {
	h := buildIndex()
	testSearch(h, "balanced")

	fmt.Printf("Saving to SIFT_small_Balanced_32M_400efc.ind\n")
	err := h.Save("SIFT_small_Balanced_32M_400efc.ind")
	assert.Nil(t, err)

	fmt.Printf("Loading from SIFT_small_Balanced_32M_400efc.ind\n")
	h2, timestamp, err := Load("SIFT_small_Balanced_32M_400efc.ind")
	assert.Nil(t, err)

	fmt.Printf("Index loaded, time saved was %v", time.Unix(timestamp, 0))

	fmt.Printf(h2.Stats())
	testSearch(h2, "balanced")
}

func TestSIFT(t *testing.T) {
	h := buildIndex()
	testSearch(h, "balanced")
}

func TestBalanced(t *testing.T) {
	// 选择使用的数据集 1是SIFT 2是SIFTsmall
	testType := 2
	if testType == 1 {
		fmt.Printf("Loading from SIFT_Balanced_32M_400efc.ind\n")
		h, timestamp, err := Load("SIFT_Balanced_32M_400efc.ind")
		assert.Nil(t, err)

		fmt.Printf("SIFT_Balanced_32M_400efc.ind loaded, time saved was %v", time.Unix(timestamp, 0))

		fmt.Printf(h.Stats())
		testSearch(h, "balanced")

		fmt.Printf("Loading from SIFT_Original_32M_400efc.ind\n")
		h2, timestamp2, err2 := Load("SIFT_Original_32M_400efc.ind")
		assert.Nil(t, err2)

		fmt.Printf("SIFT_Original_32M_400efc.ind loaded, time saved was %v", time.Unix(timestamp2, 0))

		fmt.Printf(h2.Stats())
		testSearch(h2, "origin")
		showResults()
	} else if testType == 2 {
		fmt.Printf("Loading from SIFT_small_Balanced_32M_400efc.ind\n")
		h, timestamp, err := Load("SIFT_small_Balanced_32M_400efc.ind")
		assert.Nil(t, err)

		fmt.Printf("SIFT_small_Balanced_32M_400efc.ind loaded, time saved was %v", time.Unix(timestamp, 0))

		fmt.Printf(h.Stats())
		testSearch(h, "balanced")

		fmt.Printf("Loading from SIFT_small_Original_32M_400efc.ind\n")
		h2, timestamp2, err2 := Load("SIFT_small_Original_32M_400efc.ind")
		assert.Nil(t, err2)

		fmt.Printf("SIFT_small_Original_32M_400efc.ind loaded, time saved was %v", time.Unix(timestamp2, 0))

		fmt.Printf(h2.Stats())
		testSearch(h2, "origin")
		showResults()
	}
}

func showResults() {
	originalTimeBetter := 0
	balancedTimeBetter := 0
	originalPrecisionBetter := 0
	balancedPrecisionBetter := 0

	for i, ef := range efSearch {
		fmt.Printf("\n------Comparing with ef=%v------\n", ef)
		fmt.Printf("Time in origin add: %v\n", originalTime[i])
		fmt.Printf("Time in balanced add: %v\n", balancedTime[i])
		fmt.Printf("Precision in origin add: %v\n", originalPrecision[i])
		fmt.Printf("Precision in balanced add: %v\n", balancedPrecision[i])
		if originalTime[i] > balancedTime[i] {
			balancedTimeBetter++
			balancedTimeBetterEf = append(balancedTimeBetterEf, efSearch[i])
		} else if originalTime[i] < balancedTime[i] {
			originalTimeBetter++
			originalTimeBetterEf = append(originalTimeBetterEf, efSearch[i])

		}
		if originalPrecision[i] > balancedPrecision[i] {
			originalPrecisionBetter++
			originalPrecisionBetterEf = append(originalPrecisionBetterEf, efSearch[i])
		} else if originalPrecision[i] < balancedPrecision[i] {
			balancedPrecisionBetter++
			balancedPrecisionBetterEf = append(balancedPrecisionBetterEf, efSearch[i])
		}
	}

	fmt.Printf("Time: Original add better times: %v\n", originalTimeBetter)
	fmt.Printf("Time: Balanced add better times: %v\n", balancedTimeBetter)
	fmt.Printf("Precision: Original add better times: %v\n", originalPrecisionBetter)
	fmt.Printf("Precision: Balanced add better times: %v\n\n", balancedPrecisionBetter)
	fmt.Println("Time: original add better ef")
	for _, ef := range originalTimeBetterEf {
		fmt.Printf("%v ", ef)
	}
	fmt.Println("\nTime: balanced add better ef")
	for _, ef := range balancedTimeBetterEf {
		fmt.Printf("%v ", ef)
	}
	fmt.Println("\nPrecision: original add better ef")
	for _, ef := range originalPrecisionBetterEf {
		fmt.Printf("%v ", ef)
	}
	fmt.Println("\nPrecision: balanced add better ef")
	for _, ef := range balancedPrecisionBetterEf {
		fmt.Printf("%v ", ef)
	}

	fmt.Print("\n")
	for i := 0; i < len(efSearch); i++ {
		fmt.Printf("\n------efSearch: %v------\n", efSearch[i])
		originalTimeStat := stat.Float64Slice(originalTimeEach[i])
		originalTimeMean := stat.Mean(originalTimeStat)
		originalTimeVariance := math.Sqrt(stat.Variance(originalTimeStat))
		balancedTimeStat := stat.Float64Slice(balancedTimeEach[i])
		balancedTimeMean := stat.Mean(balancedTimeStat)
		balancedTimeVariance := math.Sqrt(stat.Variance(balancedTimeStat))

		fmt.Printf("Original mean: %v\n", originalTimeMean)
		fmt.Printf("Balanced mean: %v\n", balancedTimeMean)
		fmt.Printf("Original variance: %v\n", originalTimeVariance)
		fmt.Printf("Balanced variance: %v\n", balancedTimeVariance)
	}
}

func buildIndex() *Hnsw {
	// BUILD INDEX
	var p Point
	p = make([]float32, 128)
	h := New(32, 400, p)
	h.DelaunayType = 1
	h.Grow(dataSize)

	buildStart := time.Now()
	fmt.Printf("Loading data and building index\n")
	points := make(chan job)
	go loadDataFromFvec(prefix, points)
	buildFromChan(h, points)
	buildStop := time.Since(buildStart)
	fmt.Printf("Index build in %v\n", buildStop)
	fmt.Printf(h.Stats())

	return h
}

// 记录时间
var originalTime, balancedTime = make([]float64, len(efSearch)), make([]float64, len(efSearch))

// 记录精度
var originalPrecision, balancedPrecision = make([]float64, len(efSearch)), make([]float64, len(efSearch))

//记录某种情况下add速度更佳情况下的ef
var originalTimeBetterEf, balancedTimeBetterEf = make([]int, 0), make([]int, 0)

//记录某种情况下add精度更佳情况下的ef
var originalPrecisionBetterEf, balancedPrecisionBetterEf = make([]int, 0), make([]int, 0)

// 记录每一条搜索的时间
var originalTimeEach = make([][]float64, 0)
var balancedTimeEach = make([][]float64, 0)

// var originalTimeEach = make(map[int][]float64)
// var balancedTimeEach = make(map[int][]float64)

func testSearch(h *Hnsw, addType string) {
	// SEARCH
	for i, ef := range efSearch {
		fmt.Printf("Now searching with ef=%v\n", ef)
		bestPrecision := 0.0
		bestTime := 999.0
		for j := 0; j < 10; j++ {
			start := time.Now()
			p := search(h, queries, truth, ef, addType)
			stop := time.Since(start)
			bestPrecision = math.Max(bestPrecision, p)
			bestTime = math.Min(bestTime, stop.Seconds()/float64(len(queries)))
		}
		fmt.Printf("Best Precision 10-NN: %v\n", bestPrecision)
		fmt.Printf("Best time: %v s (%v queries / s)\n", bestTime, 1/bestTime)
		if addType == "origin" {
			originalTime[i] = bestTime
			originalPrecision[i] = bestPrecision
		} else if addType == "balanced" {
			balancedTime[i] = bestTime
			balancedPrecision[i] = bestPrecision
		}
	}
}

type job struct {
	p  Point
	id uint32
}

func buildFromChan(h *Hnsw, points chan job) {
	var wg sync.WaitGroup
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func() {
			for {
				job, more := <-points
				if !more {
					wg.Done()
					return
				}
				// h.Add(job.p, job.id)
				h.BalancedAdd(job.p, job.id)
			}
		}()
	}
	wg.Wait()
}

func search(h *Hnsw, queries []Point, truth [][]uint32, efSearch int, addType string) float64 {
	var p int32
	var wg sync.WaitGroup
	l := runtime.NumCPU()
	b := len(queries) / l

	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func(queries []Point, truth [][]uint32) {
			timeRecord := make([]float64, 0)
			for j := range queries {
				startSearch := time.Now()
				results := h.Search(queries[j], efSearch, 10)
				stopSearch := time.Since(startSearch).Seconds()
				timeRecord = append(timeRecord, stopSearch * 1000)
				// calc 10-NN precision
				for results.Len() > 10 {
					results.Pop()
				}
				for _, item := range results.Items() {
					for k := 0; k < 10; k++ {
						// !!! Our index numbers starts from 1
						if int32(truth[j][k]) == int32(item.ID)-1 {
							atomic.AddInt32(&p, 1)
						}
					}
				}
			}
			if addType == "origin" {
				originalTimeEach = append(originalTimeEach, timeRecord)
				// originalTimeEach[efSearch] = timeRecord
			} else if addType == "balanced" {
				balancedTimeEach = append(balancedTimeEach, timeRecord)
				// balancedTimeEach[efSearch] = timeRecord
			}
			wg.Done()
		}(queries[i*b:i*b+b], truth[i*b:i*b+b])
	}
	wg.Wait()
	return (float64(p) / float64(10*b*l))
}

func readFloat32(f *os.File) (float32, error) {
	bs := make([]byte, 4)
	_, err := f.Read(bs)
	return float32(math.Float32frombits(binary.LittleEndian.Uint32(bs))), err
}

func readUint32(f *os.File) (uint32, error) {
	bs := make([]byte, 4)
	_, err := f.Read(bs)
	return binary.LittleEndian.Uint32(bs), err
}

func loadQueriesFromFvec(prefix string) (queries []Point, truth [][]uint32) {
	f2, err := os.Open(prefix + "_query.fvecs")
	if err != nil {
		panic("couldn't open query data file")
	}
	defer f2.Close()
	queries = make([]Point, 10000)
	qcount := 0
	for {
		d, err := readUint32(f2)
		if err != nil {
			break
		}
		if d != 128 {
			panic("Wrong dimension for this test...")
		}
		queries[qcount] = make([]float32, 128)
		for i := 0; i < int(d); i++ {
			queries[qcount][i], err = readFloat32(f2)
		}
		qcount++
	}
	queries = queries[0:qcount] // resize it
	fmt.Printf("Read %v query records\n", qcount)
	fmt.Printf("Loading groundtruth\n")
	// load query Vectors
	f3, err := os.Open(prefix + "_groundtruth.ivecs")
	if err != nil {
		panic("couldn't open groundtruth data file")
	}
	defer f3.Close()
	truth = make([][]uint32, 10000)
	tcount := 0
	for {
		d, err := readUint32(f3)
		if err != nil {
			break
		}
		if d != 100 {
			panic("Wrong dimension for this test...")
		}
		vec := make([]uint32, d)
		for i := 0; i < int(d); i++ {
			vec[i], err = readUint32(f3)
		}
		truth[tcount] = vec
		tcount++
	}
	fmt.Printf("Read %v truth records\n", tcount)

	if tcount != qcount {
		panic("Count mismatch queries <-> groundtruth")
	}

	return queries, truth
}

func loadDataFromFvec(prefix string, points chan job) {
	f, err := os.Open(prefix + "_base.fvecs")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	count := 1
	for {
		d, err := readUint32(f)
		if err != nil {
			break
		}
		if d != 128 {
			panic("Wrong dimension for this test...")
		}
		var vec Point
		vec = make([]float32, 128)
		for i := 0; i < int(d); i++ {
			vec[i], err = readFloat32(f)
		}
		points <- job{p: vec, id: uint32(count)}
		count++
		if count%1000 == 0 {
			fmt.Printf("Read %v records\n", count)
		}
	}
	close(points)
}
