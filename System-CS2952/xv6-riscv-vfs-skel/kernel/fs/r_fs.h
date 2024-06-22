#pragma once

#include "vfs.h"

void init();

struct super_block *xv6_mount(const char *source) ;

int xv6_umount(struct super_block *sb);

struct inode *xv6_alloc_inode(struct super_block *sb, short type);

void xv6_write_inode(struct inode *ino);

void xv6_free_inode(struct inode *ino);

void xv6_release_inode(struct inode *ino);

void xv6_trunc(struct inode *ino) ;

struct file *xv6_open(struct inode *ino, unsigned int mode);

void xv6_close(struct file *f);

int xv6_read(struct inode *ino, char dst_is_user, uint64 dst, unsigned int off, unsigned int n) ;

int xv6_link(struct dentry *target);

int xv6_write(struct inode *ino, char src_is_user, uint64 src, unsigned int off, unsigned int n) ;

int xv6_create(struct inode *dir, struct dentry *target, short type, short major, short minor) ;

int xv6_unlink(struct dentry *d) ;

struct dentry *xv6_dirlookup(struct inode *dir, const char *name);

void xv6_release_dentry(struct dentry *de);

int xv6_isdirempty(struct inode *dir);

void xv6_init(void) ;

void xv6_update_inode(struct inode *ip);
