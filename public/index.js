tf.loadLayersModel("unet-model-json/model.json").then(function (model){
	window.model = model
})

const submit_button = document.querySelector('.submit-button')
const new_height = 256
const new_width = 512
let image_card = document.querySelector('.image-card')
let mask_card = document.querySelector('.mark-card')
let image = document.querySelector('.image')
let dummy_image = document.querySelector('.dummy-image')

function make_prediction(preprocessed_image){
	if (window.model){
		let predicted = window.model.predict(preprocessed_image)
		let predicted_mask = create_mask(predicted)
		let mask_shape = predicted_mask.shape
		predicted_mask = predicted_mask.reshape([mask_shape[1], mask_shape[2], mask_shape[3]])
		display_mask(predicted_mask)
	}
}

function create_mask(predicted){
  const largest_index_of_predicted = tf.argMax(predicted, axis=-1)
  predicted_mask = tf.expandDims(largest_index_of_predicted, axis=-1)
  return predicted_mask
}


function display_mask(predicted_mask){
	// let mask_tag = document.querySelector('.mask')
	// let canvas = document.createElement('canvas')
	let canvas = document.querySelector('.canvas')
	canvas.height = predicted_mask.shape[0]
	canvas.width = predicted_mask.shape[1]
	// let canvas = _canvas.getContext('2d')
	// canvas.fillRect(0, 0, predicted_mask.shape[0], predicted_mask.shape[1])
	// canvas.onload = function (){ 
	// 	tf.browser.toPixels(predicted_mask, canvas)
	// }
	tf.browser.toPixels(predicted_mask, canvas)
	console.log(predicted_mask.shape)
	// tf.node.encodeJpeg(predicted_mask).then(function (s){
	// 	fs.writeFileSync("predicted_mask.jpg", s)
	// 	console.log("jpen written")
	// })
	// pixels.then(function (mask){
	// 	// mask_tag.url = mask.src
	// 	console.log(mask)
	// })
}


function preprocess_and_normalize_image(image){
	image = tf.browser.fromPixels(image).toFloat()
	
	image = tf.image.resizeNearestNeighbor(image, [new_height, new_width]).div(tf.scalar(255))
	image = tf.cast(image, dtype="float32")
	image = tf.expandDims(image, axis=0)
	return image
}


submit_button.addEventListener('click', function(event){
	let uploaded_image = document.querySelector('.uploaded-image').files[0]
	
	if (uploaded_image){

		image.src = URL.createObjectURL(uploaded_image)

		dummy_image.src = URL.createObjectURL(uploaded_image)
		
		dummy_image.onload = function(){
			let preprocessed_image = preprocess_and_normalize_image(dummy_image)
			// console.log(preprocessed_image.shape)
			// preprocessed_image.print()
			// let predicted_mask = make_prediction(preprocessed_image)
			make_prediction(preprocessed_image)
			// console.log(predicted_mask)
		}

	}
})