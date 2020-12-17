$(document).ready(function(){
	// when pressed dropdown
	var chartInterval
	var doubleChartInterval
	var currentTheme
	var currentLanguage = "English"
	var currentIncidents = []
	const months = [
		"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
	]

	// configure time header
	setInterval(function(){
		var time_string = ""
		var now = new Date()

		var hour = new String(now.getHours())
		hour = hour.length < 2 ? '0' + hour : hour

		var minute = new String(now.getMinutes())
		minute = minute.length < 2 ? '0' + minute : minute

		var day = new String(now.getDate())
		day = day.length < 2 ? '0' + day : day

		var month = months[now.getMonth()]
		var year = now.getFullYear()

		time_string = hour + ":" + minute + " - " + month + ", " + day + " " + year
		$("#time_header_").html(time_string)
	}, 1000)

	// attach english meaning to each of .translate span tag
	$(".translate").each(function(i,obj){
		$(obj).attr("meaning" ,$(obj).html())
	})

	var flash_message = function(message) {
		var x = $("<div>").attr("id", "snackbar")
			.css("box-shadow", "5px 5px black")
			.css("color", "coral")
			.css("font-family", "Courier")
			.css("font-weight", "bolder")
			.attr("class", "show")
			.html(message)
			.appendTo("#content")

		setTimeout(function(){ x.attr("class", ""); x.remove();}, 3000);
	}

	// an object of dictionaries with keys being the english words
	var dictionaries = {
		chinese : {
			'Camera View' : '相机视图',
			'Data Visualization' : '数据可视化',
			'Incident Log' : '发烧日志',
			'Settings' : '设定',
			'Languages' : '语言',
			'Camera Colorspace' : '相机色彩空间',
			'Themes' : '主题',
			'English' : '英语',
			'Chinese' : '中文',
			'Welcome to Thermal M' : '欢迎来到 ThermalM',
			'Live View' : '即时取景',
			'Dark' : '暗',
			'Light' : '光',
			'Double View' : "双重看法",
			'Download Data' : '下载资料',
			'Temperature' : '温度',
			'Temp' : '温度',
			'Distance away':  '距离',
			'Wearing mask' : '戴口罩',
			'With Mask' : '有',
			'Without Mask' : '没有',
			'Identity' : '身分',
			'Check-in' : '进入',
			'Time' : '进入',
			'Date' : '日期',
			'Number of people' : '人数',
			'Number of people in-store' : '店内人数',
			'Surrounding temperature' : '周围温度',
			'Temperature Threshold' : '温度阈值',
			'Malay' : '马来语',
			'Japanese' : '日本',
			'Metric Settings' : '公制',
			'Celcius' : '摄氏(°C)',
			'Fahrenheit' : '华氏温度(°F)',
			'Restart Server' : '重新啟動服務器',
			'Reboot' : '重啟'
		},
		malay : {
			'Camera View' : '相机视图',
			'Data Visualization' : '数据可视化',
			'Incident Log' : '发烧日志',
			'Settings' : '设定',
			'Languages' : '语言',
			'Camera Colorspace' : '相机色彩空间',
			'Themes' : '主题',
			'English' : '英语',
			'Chinese' : '中文',
			'Welcome to Thermal M' : '欢迎来到 ThermalM',
			'Live View' : '即时取景',
			'Dark' : '暗',
			'Light' : '光',
			'Double View' : "双重看法",
			'Download Data' : '下载资料',
			'Temperature' : '温度',
			'Temp' : '温度',
			'Distance away':  '距离',
			'Wearing mask' : '戴口罩',
			'With Mask' : '有',
			'Without Mask' : '没有',
			'Identity' : '身分',
			'Check-in' : '进入',
			'Time' : '进入',
			'Date' : '日期',
			'Number of people' : '人数',
			'Number of people in-store' : '店内人数',
			'Surrounding temperature' : '周围温度',
			'Temperature Threshold' : '温度阈值',
			'Malay' : '马来语',
			'Japanese' : '日本',
			'Metric Settings' : '公制',
			'Celcius' : '摄氏(°C)',
			'Fahrenheit' : '华氏温度(°F)',
			'Restart Server' : '重新啟動服務器',
			'Reboot' : '重啟'
		},
		japanese : {
			'Camera View' : 'カメラビュー',
			'Data Visualization' : 'データの可視化',
			'Incident Log' : 'インシデントログ',
			'Settings' : '設定',
			'Languages' : '言語',
			'Camera Colorspace' : 'カメラの色空間',
			'Themes' : 'テーマ',
			'English' : '英語',
			'Chinese' : '中国人',
			'Welcome to Thermal M' : 'ThermalMへようこそ',
			'Live View' : 'ライブビュー',
			'Dark' : '闇',
			'Light' : '光',
			'Double View' : "ダブルビュー",
			'Download Data' : 'データをダウンロード',
			'Temperature' : '温度',
			'Temp' : '温度',
			'Distance away':  '距離',
			'Wearing mask' : 'マスクを着用',
			'With Mask' : 'マスク付き',
			'Without Mask' : 'マスクなし',
			'Identity' : '身元',
			'Check-in' : 'チェックイン',
			'Time' : '時間',
			'Date' : '日付',
			'Number of people' : '人々の数',
			'Number of people in-store' : '店内人数',
			'Surrounding temperature' : '周囲温度',
			'Temperature Threshold' : '温度しきい値',
			'Malay' : 'マレー語',
			'Japanese' : '日本人',
			'Metric Settings' : 'メトリック',
			'Celcius' : 'セルシウス(°C)',
			'Fahrenheit' : '華氏(°F)',
			'Reboot' : 'リブート',
			'Restart Server' : 'サーバーを再起動'
		}
	}

	/*** Continuously poll for top incidents ***/
	setInterval(function(){
		// request the server
		var formData = new FormData()
		formData.append("secret_key", "HieuDepTry")

		$.ajax({
			url : "/poll_incident",
			type : 'POST',
			async : true,
			data : formData,
			contentType : false,
			processData : false,
			success : function(response){
				// retrieve the file name and read them
				var objects = JSON.parse(response)
				var filenames = objects['incident_img']
				var incident_count = objects['incident_count']
				var average_amb = objects['average_amb']

				// console.log(objects)
				$("#counter").html(incident_count)
				$("#aveamb").html(average_amb)

				if(JSON.stringify(filenames) != JSON.stringify(currentIncidents)){
					currentIncidents = filenames
					for(var i = 0; i < filenames.length; i++){
						var color = 'green'
						var img_id = 'double-view-img-' + (i + 1)
						var p_id = 'details-' + (i+1)

						var temp = filenames[i].split("/")[4].split('_')[0]

						if(temp > $("#temperature-threshold-input").val()){
							color = 'red'
							// var beep = new Audio("/static/audio/beep-06.mp3")

							// beep.play()
						}

						var time = filenames[i].split('_')[2]

						var hour = time.split('-')[0]
						var date = time.split('-')[1] + "/" + time.split("-")[2] + "/" + time.split("-")[3]

						if(currentLanguage == "English"){
							var string = "<strong><span class='translate small-header' meaning='Temperature'>Temp</span> : </strong>" + temp + "(&#176;C)<br>"
							string += "<strong><span class='translate small-header' meaning='Check-in'>Time</span> : </strong>" + hour + "<br>"
							string += "<strong><span class='translate small-header' meaning='Date'>Date</span> : </strong>" + date + "<br>"
						}else if(currentLanguage == 'Chinese'){
							var string = "<strong><span class='translate small-header' meaning='Temperature'>"+dictionaries['chinese']['Temperature']+"</span> : </strong>" + temp + "(&#176;C)<br>"
							string += "<strong><span class='translate small-header' meaning='Check-in'>"+dictionaries['chinese']['Check-in']+"</span> : </strong>" + hour + "<br>"
							string += "<strong><span class='translate small-header' meaning='Date'>" + dictionaries['chinese']['Date'] + "</span> : </strong>" + date + "<br>"
						}else if(currentLanguage == 'Malay'){
							var string = "<strong><span class='translate small-header' meaning='Temperature'>"+dictionaries['malay']['Temperature']+"</span> : </strong>" + temp + "(&#176;C)<br>"
							string += "<strong><span class='translate small-header' meaning='Check-in'>"+dictionaries['malay']['Check-in']+"</span> : </strong>" + hour + "<br>"
							string += "<strong><span class='translate small-header' meaning='Date'>" + dictionaries['malay']['Date'] + "</span> : </strong>" + date + "<br>"
						}else if(currentLanguage == 'Japanese'){
							var string = "<strong><span class='translate small-header' meaning='Temperature'>"+dictionaries['japanese']['Temperature']+"</span> : </strong>" + temp + "(&#176;C)<br>"
							string += "<strong><span class='translate small-header' meaning='Check-in'>"+dictionaries['japanese']['Check-in']+"</span> : </strong>" + hour + "<br>"
							string += "<strong><span class='translate small-header' meaning='Date'>" + dictionaries['japanese']['Date'] + "</span> : </strong>" + date + "<br>"
						}

						console.log("[INFO] Incident Polled ... ")
						$("#" + p_id).css('color', color).html(string)
						$("#" + img_id).attr('src', filenames[i])
					}
				}
			}
		})
	}, 500)

	/*** End of poll ***/

	$("#download_data").click(function(){
		// just send a request to /download
		window.open("/download", "_blank")
	})

	$("#general-mgmt").click(function(){
		if($("#general-mgmt-dropdown").css("display") == "none"){
			$("#arrow-general-mgmt").html("&#9660")
		}else{
			$("#arrow-general-mgmt").html("&#9654")
		}

		$("#general-mgmt-dropdown").slideToggle(400)
	})

	$("#settings-mgmt").click(function(){
		if($("#settings-mgmt-dropdown").css("display") == "none"){
			$("#arrow-settings-mgmt").html("&#9660")
		}else{
			$("#arrow-settings-mgmt").html("&#9654")
		}

		$("#settings-mgmt-dropdown").slideToggle(400)
	})

	var amb_chart 
	var pph_chart

	var plot = function(canvas, data_, x_data, type){
		let range = n => [...Array(n).keys()]

		var label_ = ""
		var title = ""

		if(type == "amb"){
			label_ = "Temperature (Celcius degree)"
			title = "Average surrouding temperature"

			if(amb_chart != null){
				amb_chart.destroy()
			}

			// var x_data = range(data_.length)
			amb_chart = new Chart(canvas, {
			  type: 'line',
			  data: {
				labels: x_data,
				datasets: [{ 
					data: data_,
					label: label_,
					borderColor: "#3e95cd",
					fill: false
				  }]
			  },
			  options: {
				responsive : true,
				title: {
				  display: true,
				  text: title
				},
				scales: {
					  
					  xAxes: [{
						ticks: {
						  maxTicksLimit: 10,
						  maxRotations : 60,
						  minRotations : 60
						}
					  }]
				}
			  }
			});
		}else{
			label_ = "People per hour"
			title = "Pass-by frequency"

			if(pph_chart != null){
				pph_chart.destroy()
			}

			// var x_data = range(data_.length)
			pph_chart = new Chart(canvas, {
			  type: 'line',
			  data: {
				labels: x_data,
				datasets: [{ 
					data: data_,
					label: label_,
					borderColor: "#3e95cd",
					fill: false
				  }]
			  },
			  options: {
				responsive : true,
				title: {
				  display: true,
				  text: title
				},
				scales: {
					  
					  xAxes: [{
						ticks: {
						  maxTicksLimit: 10,
						  maxRotations : 60,
						  minRotations : 60
						}
					  }]
				}
			  }
			});
		}
	}

	$("#data_visualization").click(function(){
		$(".box").slideUp("fast")
		$("#main-table").attr('border', 0)
		$("#double-view-incidents-region").slideUp("fast")
		$("#incident_list").slideUp("fast")

		$("#chart-region").slideDown("fast")
		var amb_chart = document.getElementById("amb-temp-list")
		var people_chart = document.getElementById("people-per-hour-list")
		var formData = new FormData()
		formData.append("secret_key", "HieuDepTry")

		// first time plotting
		$.ajax({
			url : '/get_data',
			type : 'POST',
			async : true,
			data : formData,
			processData : false,
			contentType : false,
			success : function(response){
				var plot_data = JSON.parse(response)

				var data = plot_data[0]
				var axis = plot_data[1]

				var amb_data = data[0]
				var pph_data = data[1]

				plot(amb_chart, amb_data, axis[0], "amb")
				plot(people_chart, pph_data, axis[1], "pph")
			}
		})
		chartInterval = setInterval(function(){
			// send request to the server to get data
			$.ajax({
				url : '/get_data',
				type : 'POST',
				async : true,
				data : formData,
				processData : false,
				contentType : false,
				success : function(response){
					var plot_data = JSON.parse(response)

					var data = plot_data[0]
					var axis = plot_data[1]

					var amb_data = data[0]
					var pph_data = data[1]

					plot(amb_chart, amb_data, axis[0], "amb")
					plot(people_chart, pph_data, axis[1], "pph")
				}
			})


		}, 60 * 1 * 1000)
	})

	$("#camera_view").click(function(){
		clearInterval(chartInterval)
		$("#chart-region").slideUp("fast")
		$("#incident_list").slideUp("fast")
		$("#double-view-incidents-region").slideDown("fast")

		$("#main-table").attr("border", 2)
		$(".box").slideDown("fast")
	})

	$("#incident_log").click(function(){
		$("#chart-region").slideUp("fast")
		$("#double-view-incidents-region").slideUp("fast")
		$(".box").slideUp("fast")
		$("#main-table").attr('border', 0)

		// clear the incident list
		$("#incident_list").empty()

		var formData = new FormData()
		formData.append("secret_key", "HieuDepTry")

		var text_color = 'white'
		if(currentTheme == 'light'){
			text_color = 'black'
		}

		$.ajax({
			url : '/get_incidents',
			type : 'POST',
			async : true, 
			data : formData,
			processData : false,
			contentType : false,
			success : function(response){
				// the server will respond with a list of files in the
				// incident logs
				var base_dir = '/static/img/incidents/'
				var files = JSON.parse(response)

				for (var i = 0; i < files.length; i++){
					var abs_path = base_dir + files[i]

					var details = files[i].split("_")
					var temperature = details[0]
					var distance = details[1]
					//var mask = details[2]
					//var identity = details[3]
					var checkin = details[2]

					if(currentLanguage == 'English'){
						var string = "<strong><span class='translate' meaning='Temperature'>Temperature</span> : </strong>" + temperature + " (&#176;C)<br>"
						string += "<strong><span class='translate' meaning='Distance away'>Distance away</span> : </strong>" + distance + " (cm) <br>"
						string += "<strong><span class='translate' meaning='Check-in'>Check-in</span> : </strong>" + checkin + "<br>" 
					}else if(currentLanguage == 'Chinese'){
						var string = "<strong><span class='translate' meaning='Temperature'>" + dictionaries['chinese']['Temperature'] + "</span> : </strong>" + temperature + " (&#176;C)<br>"
						string += "<strong><span class='translate' meaning='Distance away'>" + dictionaries['chinese']['Distance away'] + "</span> : </strong>" + distance + " (cm) <br>"						
						string += "<strong><span class='translate' meaning='Check-in'>" + dictionaries['chinese']['Check-in'] + "</span> : </strong>" + checkin + "<br>" 
					}else if(currentLanguage == 'Malay'){
						var string = "<strong><span class='translate' meaning='Temperature'>" + dictionaries['malay']['Temperature'] + "</span> : </strong>" + temperature + " (&#176;C)<br>"
						string += "<strong><span class='translate' meaning='Distance away'>" + dictionaries['malay']['Distance away'] + "</span> : </strong>" + distance + " (cm) <br>"						
						string += "<strong><span class='translate' meaning='Check-in'>" + dictionaries['malay']['Check-in'] + "</span> : </strong>" + checkin + "<br>" 
					}else if(currentLanguage == 'Japanese'){
						var string = "<strong><span class='translate' meaning='Temperature'>" + dictionaries['japanese']['Temperature'] + "</span> : </strong>" + temperature + " (&#176;C)<br>"
						string += "<strong><span class='translate' meaning='Distance away'>" + dictionaries['japanese']['Distance away'] + "</span> : </strong>" + distance + " (cm) <br>"						
						string += "<strong><span class='translate' meaning='Check-in'>" + dictionaries['japanese']['Check-in'] + "</span> : </strong>" + checkin + "<br>" 
					}
					$("<li>")
						.css("box-shadow", "2px 2px #00002280")
						.addClass("incident_list_item")
						.attr("id", "incident_id_" + i)
						.appendTo("#incident_list")

					$("<img>")
						.css("display", "inline-block")
						.attr("src", abs_path)
						.appendTo("#incident_id_" + i)

					$("<p>")
						.html(string)
						.addClass("incident-log-info-text")
						.css("color", text_color)
						.css("font-size", "20px")
						.css("margin-left", "20px")
						.css("display", "inline-block")
						.appendTo("#incident_id_" + i)
				}

				$("#incident_list").slideDown("fast")
			}
		})
	})

	$("#language_setting").click(function(){
		if($("#setting-language-sub-menu").css("display") == "none"){
			$("#language_arrow").html("&#9660")
		}else{
			$("#language_arrow").html("&#9654")
		}

		$("#setting-language-sub-menu").slideToggle(400)
	})

	$("#set-chinese").click(function(){
		// set all translatable phrases to Chinese
		currentLanguage = 'Chinese'
		$(".translate").each(function(i,obj){
			$(obj).html(dictionaries['chinese'][$(obj).attr("meaning")])
		})

		// send this language to the server 
		var formData = new FormData()
		formData.append('language', currentLanguage)

		$.ajax({
			url : "/set_language",
			type : "POST",
			data : formData,
			async : true,
			processData : false,
			contentType : false,
			success : function(response){
				flash_message("Language set to Chinese")
			}
		})
		// console.log($(".translate").html())
	})

	$("#set-malay").click(function(){
		// set all translatable phrases to Chinese
		currentLanguage = 'Malay'
		$(".translate").each(function(i,obj){
			$(obj).html(dictionaries['malay'][$(obj).attr("meaning")])
		})
		// console.log($(".translate").html())
		// send this language to the server 
		var formData = new FormData()
		formData.append('language', currentLanguage)

		$.ajax({
			url : "/set_language",
			type : "POST",
			data : formData,
			async : true,
			processData : false,
			contentType : false,
			success : function(response){
				flash_message("Language set to Malay")
			}
		})
	})

	$("#set-japan").click(function(){
		// set all translatable phrases to Chinese
		currentLanguage = 'Japanese'
		$(".translate").each(function(i,obj){
			$(obj).html(dictionaries['japanese'][$(obj).attr("meaning")])
		})
		console.log("[INFO] Language set to Chinese ... ")
		// console.log($(".translate").html())


		// send this language to the server 
		var formData = new FormData()
		formData.append('language', currentLanguage)

		$.ajax({
			url : "/set_language",
			type : "POST",
			data : formData,
			async : true,
			processData : false,
			contentType : false,
			success : function(response){
				flash_message("Language set to Japanese")
			}
		})
	})

	$("#set-english").click(function(){
		currentLanguage = 'English'
		
		$(".translate").each(function(i, obj){
			$(obj).html($(obj).attr("meaning"))
		})

		// send this language to the server 
		var formData = new FormData()
		formData.append('language', currentLanguage)

		$.ajax({
			url : "/set_language",
			type : "POST",
			data : formData,
			async : true,
			processData : false,
			contentType : false,
			success : function(response){
				flash_message("Language set to English")
			}
		})
	})

	$("#camera_setting").click(function(){
		// just gonna show the menu
		if($("#setting-camera-sub-menu").css("display") == "none"){
			$("#camera_arrow").html("&#9660")
		}else{
			$("#camera_arrow").html("&#9654")
		}

		$("#setting-camera-sub-menu").slideToggle(400)
	})

	$("#metric_setting").click(function(){
		// just gonna show the menu
		if($("#setting-metric-sub-menu").css("display") == "none"){
			$("#metric_arrow").html("&#9660")
		}else{
			$("#metric_arrow").html("&#9654")
		}

		$("#setting-metric-sub-menu").slideToggle(400)
	})

	$("#threshold_setting").click(function(){
		if($("#setting-threshold-sub-menu").css("display") == "none"){
			$("#threshold_arrow").html("&#9660")			
		}else{
			$("#threshold_arrow").html("&#9654")	
		}

		// need to fill in the current threshold
		var formData = new FormData()
		formData.append("secret_key", "HieuDepTry")
		$.ajax({
			type : 'POST',
			url : '/get_thresh',
			data : formData,
			async : true,
			processData : false,
			contentType : false, 
			success : function(response){
				$("#temperature-threshold-input").val(response)
			}
		})

		$("#setting-threshold-sub-menu").slideToggle(400)
	})

	$("#temperature-threshold-input").change(function(){
		var formData = new FormData()
		formData.append('secret_key', 'HieuDepTry')

		var temperature = $("#temperature-threshold-input").val()
		console.log("[INFO] Temperature threshold changed to " + temperature)
		formData.append('thresh', temperature)

		$.ajax({
			url : '/set_thresh',
			type : 'POST',
			data : formData,
			async : true,
			contentType : false,
			processData : false,
			success : function(response){

			}
		})
	})

	$("#set-gray").click(function(){
		// just send a request to the flask server
		// the flask server will send a signal to the camera to make
		// it change the thermal cam to grayscale
		
		var formData = new FormData()
		formData.append("camera_view", "GRAY")

		$.ajax({
			url : "/change_camera_view",
			type : 'POST',
			async : true, 
			data : formData,
			processData : false,
			contentType : false,
			success : function(response){
				flash_message("Camera colorspace set to Gray Scale")
			}
		})
	})

	$("#set-rgb").click(function(){
		// same thing as set gray
		// just a normal request to flask server
		
		var formData = new FormData()
		formData.append("camera_view", "RGB")

		$.ajax({
			url : "/change_camera_view",
			type : "POST",
			async : true,
			data : formData,
			processData : false,
			contentType : false,
			success : function(response){
				flash_message("Camera colorspace set to RGB")
			}
		})
	})

	$("#theme_setting").click(function(){
		// just gonna show the menu
		if($("#setting-theme-sub-menu").css("display") == "none"){
			$("#theme_arrow").html("&#9660")
		}else{
			$("#theme_arrow").html("&#9654")
		}

		$("#setting-theme-sub-menu").slideToggle(400)
	})

	$("#set-dark").click(function(){
		currentTheme = 'dark'
		$("body").css("background-color", "#121212")
		$(".incident-log-info-text").css("color", "white")

		$(".navbar").addClass("navbar-dark")
		$(".navbar").addClass("bg-dark")

		$(".navbar").removeClass("navbar-light")
		$(".navbar").removeClass("bg-light")
	})

	$("#set-light").click(function(){
		currentTheme = 'light'
		$("body").css("background-color", "white")
		$(".incident-log-info-text").css("color", "black")

		$(".navbar").addClass("navbar-light")
		$(".navbar").addClass("bg-light")

		$(".navbar").removeClass("navbar-dark")
		$(".navbar").removeClass("bg-dark")
	})

	$("#set-celcius").click(function(){
		var formData = new FormData()
		formData.append('metric', 'Celcius')

		$.ajax({
			url : '/set_metric',
			type : 'POST',
			data : formData,
			async : true,
			contentType : false,
			processData : false,
			success : function(response){
				flash_message("Metric set to Celcius")
			}
		})
	})

	$("#set-fahrenheit").click(function(){
		var formData = new FormData()
                formData.append('metric', 'Fahrenheit')

                $.ajax({
                        url : '/set_metric',
                        type : 'POST',
                        data : formData,
                        async : true,
                        contentType : false,
                        processData : false,
                        success : function(response){
                            flash_message("Metric set to Fahrenheit")
                        }
                })

	})

	// restarting and shutting down machine 
	$("#restart_server").click(function(){
		var shutdown = confirm("Do you want to shutdown this machine ?")

		if(shutdown){
			alert("THe machine will shutdown shortly ... ")
			var formData = new FormData()
			$.ajax({
				url : "/shutdown",
				type : "POST",
				data : formData,
				async : true,
				contentType : false,
				processData : false,
				success : function(response){
					console.log("[INFO] Server is restarting ... ")
				}
			})
		}
	})

	$("#reboot").click(function(){
		var reboot = confirm("Do you want to restart this machine ?")
		if(reboot){
			var formData = new FormData()
			$.ajax({
				url : "/reboot",
				type : "POST",
				data : formData,
				async : true,
				contentType : false,
				processData : false,
				success : function(response){
					console.log("[INFO] Server is rebooting ... ")
				}
			})
		}
	})
})
