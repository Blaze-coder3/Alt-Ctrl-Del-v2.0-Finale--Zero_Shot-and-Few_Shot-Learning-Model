In order to setup this zero-shot GroundingDino

1)clone the GroundingDino repositry from Github.

2)Install all the libraries in requirements file using pip.

3)cd GroundingDino to change its current working directory.

4) follow these steps one after the other 

	%cd {HOME}/GroundingDINO

	git checkout -q 57535c5a79791cb76e36fdb64975271354f10251

	pip install -q -e.

	wget https://github.com/IDEA-Research/GroundingDINO/releases/download/vo.1.0-alpha/groundingdino_swint_ogc.pth

5) so that you can extract the features from Groundingdino
6) web.py is our final model