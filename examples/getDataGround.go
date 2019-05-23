package main

import (
	hnsw ".."
	"bufio"
	"fmt"
	"github.com/grd/stat"
	"io"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type job struct {
	p    []float32
	attr []string
	id   uint32
}

type query struct {
	p    []float32
	attr []string
}

const (
	// sift small
	NUM2     = 10000
	TESTNUM2 = 100
	efSearch = 200
	K        = 100

	// sift
	//NUM2      = 1000000
	//TESTNUM2  = 10000
	//efSearch = 1000
	//K        = 1000
	DIMENSION2 = 128

	// gist
	//NUM2           = 1000000
	//TESTNUM2       = 1000
	//efSearch       = 1000
	//K              = 1000
	//DIMENSION2     = 960

	M              = 16
	efConstruction = 400
)

func main() {
	//preType := "gist"
	preType := "siftsmall"

	prefix := "../dataset/" + preType + "_ma/" + preType

	points := make(chan job)
	queries := make(chan job)

	querySlice := make([]query, TESTNUM2)

	go loadBaseData(prefix, points)
	go loadQueryData(prefix, queries)

	p := make([]float32, DIMENSION2)
	h := hnsw.New(M, efConstruction, p)
	h.Grow(NUM2)

	var wg sync.WaitGroup
	for i := 0; i < runtime.NumCPU()-4; i++ {
		wg.Add(1)
		go func() {
			for {
				job, more := <-points
				if !more {
					wg.Done()
					return
				}
				h.Add(job.p, job.id, job.attr)
			}
		}()
	}
	wg.Wait()

	err := h.Save(preType + "_" + strconv.FormatInt(M, 10) + "_" + strconv.FormatInt(efConstruction, 10) + ".ind")
	if err != nil {
		panic("Save error!")
	}

	h, timestamp, _ := hnsw.Load(preType + "_" + strconv.FormatInt(M, 10) + "_" + strconv.FormatInt(efConstruction, 10) + ".ind")
	fmt.Printf("Index loaded, time saved was %v\n", time.Unix(timestamp, 0))

	//for i := 1; i <= NUM2; i++ {
	//	h.Add(points[i-1].p, uint32(i), points[i-1].attr)
	//	if (i)%1000 == 0 {
	//		fmt.Printf("%v points added\n", i)
	//	}
	//}
	//fmt.Println(h.GetNodes()[0])

	fmt.Printf("Now searching with HNSW...\n")
	timeRecord := make([]float64, TESTNUM2)
	hits := 0

	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")

	type truthTyoe struct {
		sync.RWMutex
		p [][]uint32
	}

	var truth truthTyoe
	truth.p = make([][]uint32, TESTNUM2)

	var wg2 sync.WaitGroup
	for i := 0; i < runtime.NumCPU(); i++ {
		wg2.Add(1)
		go func() {
			for {
				job, more := <-queries
				if !more {
					wg2.Done()
					return
				}
				result := h.SearchBrute(job.p, K, job.attr)
				truth.Lock()
				truth.p[int(job.id)] = make([]uint32, K)
				for j := K - 1; j >= 0; j-- {
					item := result.Pop()
					//fmt.Println(len(truth.p[int(job.id)]))

					truth.p[int(job.id)][j] = item.ID
				}
				truth.Unlock()
				//fmt.Println(truth)
				querySlice[int(job.id)] = query{p: job.p, attr: job.attr}
			}
		}()
	}
	wg2.Wait()

	//fmt.Println(truth)

	//for i := range queries {
	//	if (i)%10 == 0 {
	//		fmt.Printf("Calculating using bruteforce search: %v\n", i)
	//	}
	//	result := h.SearchBrute(queries[i].p, K, queries[i].attr)
	//	truth[i] = make([]uint32, K)
	//	for j := K - 1; j >= 0; j-- {
	//		item := result.Pop()
	//		truth[i][j] = item.ID
	//	}
	//}

	// Save ground truth to file
	var fileTruth = prefix + "_groundtruth.txt"
	f, _ := os.Create(fileTruth)
	for _, line := range truth.p {
		for i, v := range line {
			_, _ = io.WriteString(f, strconv.FormatUint(uint64(v), 10))
			if i < len(line)-1 {
				_, _ = io.WriteString(f, " ")
			}
		}
		_, _ = io.WriteString(f, "\n")
	}

	for i := 0; i < TESTNUM2; i++ {
		startSearch := time.Now()
		result := h.Search(querySlice[i].p, efSearch, K, querySlice[i].attr)
		//fmt.Print("Searching with attributes:")
		//fmt.Println(attrQuery[i])
		stopSearch := time.Since(startSearch)
		timeRecord[i] = stopSearch.Seconds() * 1000
		if result.Size != 0 {
			for j := 0; j < K; j++ {
				item := result.Pop()
				//fmt.Printf("%v  ", item)
				if item != nil {
					//fmt.Println(h.GetNodeAttr(item.ID))
					var flag = 0
					for k := 0; k < K; k++ {
						if item.ID == truth.p[i][k] {
							hits++
							flag = 1
							break
						}
					}
					if flag == 0 {
						fmt.Printf("Can't match: %v, i: %v, attr: %v", item.ID, i, querySlice[i].attr)
						fmt.Println()
					}
				}
			}
		} else {
			fmt.Println("Can't return any node")
		}

		//fmt.Println()
	}

	data := stat.Float64Slice(timeRecord)
	mean := stat.Mean(data)
	variance := stat.Variance(data)

	fmt.Printf("Mean of queries time(MS): %v\n", mean)
	fmt.Printf("Variance of queries time: %v\n", variance)
	fmt.Printf("%v queries / second (single thread)\n", 1000.0/mean)
	fmt.Printf("Average %v-NN precision: %v\n", K, float64(hits)/(float64(TESTNUM2)*float64(K)))
	fmt.Printf("\n")
	fmt.Printf(h.Stats())

}

func loadBaseData(prefix string, points chan job) {
	f, err := os.Open(prefix + "_base.txt")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	count := 1
	for s.Scan() {
		list := strings.Split(s.Text(), " ")
		vec := make([]float32, DIMENSION2)
		attr := list[DIMENSION2:]
		for i := 0; i < int(DIMENSION2); i++ {
			v, _ := strconv.ParseFloat(list[i], 32)
			vec[i] = float32(v)
		}
		points <- job{p: vec, attr: attr, id: uint32(count)}
		count++
		if count%1000 == 0 {
			fmt.Printf("Read %v records\n", count)
		}
	}
	close(points)
}

func loadQueryData(prefix string, queries chan job) {
	f, err := os.Open(prefix + "_query.txt")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	count := 0
	for s.Scan() {
		list := strings.Split(s.Text(), " ")
		vec := make([]float32, DIMENSION2)
		attr := list[DIMENSION2:]
		for i := 0; i < int(DIMENSION2); i++ {
			v, _ := strconv.ParseFloat(list[i], 32)
			vec[i] = float32(v)
		}
		queries <- job{p: vec, attr: attr, id: uint32(count)}
		count++
		if count%1000 == 0 {
			fmt.Printf("Read %v query records\n", count)
		}
	}
	close(queries)
	return
}
