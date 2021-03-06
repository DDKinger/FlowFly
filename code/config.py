import json
import os

class Config:
  """Configuration object loaded from/saved to JSON object
  """
  def __init__(self, params):
    entries = {}
    for param in params:
      val = params[param]
      if val is not None:
        entries[param] = val
    self.__dict__.update(entries)

