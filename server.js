const express = require('express')

const path = require('path')

const app = express()

const tf = require('@tensorflow/tfjs')

app.set('view engine', 'ejs')

let PORT = process.env.PORT || 5000

app.use(express.static('public'))

app.set('views', path.join(__dirname, '/views'))

app.use(express.json())

app.use(express.urlencoded({ extended: false }))

app.get('/', function(request, response){
	return response.render('index', { title: "Semantic Image Segmentation" })
})

app.get('/notebook', function(request, response){
	return response.render('notebook', { title: "Notebook: Semantic Image Segmentation" })
})

app.listen(PORT, function(){
	console.log('server running')
})