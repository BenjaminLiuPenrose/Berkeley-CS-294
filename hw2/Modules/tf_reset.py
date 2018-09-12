# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
Name: Beier (Benjamin) Liu
Date: 9/10/2018
'''

'''===================================================================================================
File content:
function tf_reset()
==================================================================================================='''

def tf_reset():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()
