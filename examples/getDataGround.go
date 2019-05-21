package main

import (
	"io/ioutil"
	"fmt"
)

func main() {
	preType := "siftsmall"
	prefix := "./" + preType + "_ma/" + preType

	data, err := ioutil.ReadFile(prefix + "_base.fvecs")
	if err != nil {
		fmt.Println("File reading error", err)
		return
	}



}

