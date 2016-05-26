#ifndef __FILES_H
#define __FILES_H

#include <dirent.h>
#include <string>
#include <vector>

std::vector<std::string> openDirectory(std::string path = ".") {

	DIR* dir;
	dirent* pdir;
	std::vector<std::string> files;

	dir = opendir(path.c_str());

	while (pdir = readdir(dir)) {
		std::string filename(pdir->d_name);
		std::string fileExtension("mtx");

		//debug(filename);

		if (filename.find(fileExtension) == std::string::npos)
			continue;

		if ((filename.compare(".") != 0) && (filename.compare("..") != 0))
			files.push_back(filename);
	}

	return files;
}

#endif
