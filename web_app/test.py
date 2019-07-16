#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:59:22 2019

@author: seth
"""

# importing modules 
from flask import Flask, render_template 
  
# declaring app name 
app = Flask(__name__) 
  
# making list of pokemons 
Pokemons =["Pikachu", "Charizard", "Squirtle", "Jigglypuff",  
           "Bulbasaur", "Gengar", "Charmander", "Mew", "Lugia", "Gyarados"] 
  
# defining home page 
@app.route('/') 
def homepage(): 
  
# returning index.html and list 
# and length of list to html page 
    return render_template("index.html", len = len(Pokemons), Pokemons = Pokemons) 
  
    # if __name__ == '__main__': 
  
    # running app 
    app.run(use_reloader = True, debug = True) 
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
