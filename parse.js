const fs = require('fs-extra');

const walk = (dir) => {
	fs.readdir(dir, (err, list) => {
		list.forEach((dir) => {
			fs.readdir(`${__dirname}/lfw/${dir}`, (err, files) => {
				if(files.length >= 10) {
					fs.copy(`${__dirname}/lfw/${dir}`, `${__dirname}/lwf10/${dir}`, err => {
						if (err) return console.log(err);
					});
				}
			});
		});
	});
}

walk(`${__dirname}/lfw`);

